// ---------------------------------------------------------------------------
// Dummy simulation experiment
//
// Generates a 5x5 maze (11x11 grid), places a single seed node at the entry,
// and runs the simulation for NUM_STEPS steps using a freshly initialised
// (random-weight) NeuralNetwork.  Results are written to sim_output.json.
//
// Build & run:
//   cmake --build cmake-build-debug && ./cmake-build-debug/mycelium
//
// Then load sim_output.json in the HTML visualiser.
// ---------------------------------------------------------------------------

#include "node_nn/nn.h"
#include "node_nn/utils/io.h"
#include "graph.h"    // sim::Graph, sim::step, sim::cleanup_dead
#include "maze.h"     // sim::generate_maze, sim::Maze
#include "export.h"   // sim::SimExporter
#include "config.h"   // sim::load_config

#include <iostream>
#include <filesystem>
#include <vector>

int main(int argc, char* argv[]) {
    // Ensure output is not buffered
    std::cout.setf(std::ios::unitbuf);
    
    // ---- Load hyperparameters from file -------------------------------
    // Check for command-line argument first
    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
        std::cout << "Using config file from argument: " << config_path << "\n";
    } else {
        // Try both locations: project root (when run from root) and parent dir (when run from build dir)
        std::vector<std::string> config_paths = {
            "hyperparameters.txt",           // if running from project root
            "../hyperparameters.txt"         // if running from cmake-build-debug
        };
        
        bool found = false;
        for (const auto& path : config_paths) {
            if (std::filesystem::exists(path)) {
                config_path = path;
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cout << "Config file not found in default locations, using defaults.\n";
            sim::set_default_config();
            config_path = "";
        }
    }
    
    if (!config_path.empty()) {
        std::cout << "Loading config from: " << config_path << "\n";
        if (!sim::load_config(config_path)) {
            std::cout << "Failed to load config, using defaults.\n";
            sim::set_default_config();
        }
    }
    
    // ---- Maze setup ---------------------------------------------------
    constexpr int MAZE_COLS = 5;   // "room" columns  -> grid width  = 11
    constexpr int MAZE_ROWS = 5;   // "room" rows     -> grid height = 11
    constexpr unsigned MAZE_SEED = 49u;

    sim::Maze maze = sim::generate_maze(MAZE_COLS, MAZE_ROWS, MAZE_SEED);
    std::cout << "Maze: " << maze.width << " x " << maze.height << " cells\n";

    // Entry: enclosed start point at cell (1, 1) -> center (1.5, 1.5)
    // Exit:  enclosed end point at cell (W-2, H-2) -> center (W-1.5, H-1.5)
    const sim::Vec2 start  = { 1.5f, 1.5f };
    const sim::Vec2 target = { static_cast<float>(maze.width)  - 1.5f,
                               static_cast<float>(maze.height) - 1.5f };

    std::cout << "Start:  (" << start.x  << ", " << start.y  << ")\n";
    std::cout << "Target: (" << target.x << ", " << target.y << ")\n";

    // ---- Initial graph: nodes on all empty cells ----------------------
    sim::Graph graph;
    {
        const int start_col = 1;
        const int start_row = 1;
        const int end_col = maze.width - 2;
        const int end_row = maze.height - 2;

        std::vector<std::vector<int>> cell_to_node(
            maze.height,
            std::vector<int>(maze.width, -1));

        for (int row = 0; row < maze.height; ++row) {
            for (int col = 0; col < maze.width; ++col) {
                if (maze.grid[row][col] != 0) {
                    continue;
                }

                sim::Node node;
                node.pos = {sim::cell_cx(col), sim::cell_cy(row)};
                node.is_dead = false;
                node.is_source = (col == start_col && row == start_row) ||
                                 (col == end_col && row == end_row);
                node.is_pinned = node.is_source;
                node.energy = node.is_source ? sim::ENERGY_SOURCE_VALUE : sim::ENERGY_INITIAL;

                graph.nodes.push_back(node);
                cell_to_node[row][col] = static_cast<int>(graph.nodes.size()) - 1;
            }
        }

        auto connect_if_open = [&](int row_a, int col_a, int row_b, int col_b) {
            if (row_b < 0 || row_b >= maze.height || col_b < 0 || col_b >= maze.width) {
                return;
            }

            int idx_a = cell_to_node[row_a][col_a];
            int idx_b = cell_to_node[row_b][col_b];
            if (idx_a < 0 || idx_b < 0) {
                return;
            }

            graph.nodes[idx_a].edges.push_back({idx_b, sim::INITIAL_WEIGHT});
            graph.nodes[idx_b].edges.push_back({idx_a, sim::INITIAL_WEIGHT});
        };

        for (int row = 0; row < maze.height; ++row) {
            for (int col = 0; col < maze.width; ++col) {
                if (cell_to_node[row][col] < 0) {
                    continue;
                }
                connect_if_open(row, col, row, col + 1);
                connect_if_open(row, col, row + 1, col);
            }
        }

        std::cout << "Initial graph: " << graph.nodes.size() << " nodes (all empty cells), start/end pinned\n";
    }

    // ---- Neural network: load trained model --------------------------
    node_nn::NeuralNetwork nn;
    const std::vector<std::string> model_paths = {
        "node_nn_model.nn",     // if running from project root
        "../node_nn_model.nn"   // if running from cmake-build-debug
    };

    std::string loaded_model_path;
    for (const auto& model_path : model_paths) {
        if (node_nn::load_model(model_path, nn)) {
            loaded_model_path = model_path;
            break;
        }
    }

    if (loaded_model_path.empty()) {
        std::cerr << "Error: Could not load trained model node_nn_model.nn\n";
        return 1;
    }

    std::cout << "NeuralNetwork: loaded trained model from " << loaded_model_path << "\n";

    // ---- Run simulation ----------------------------------------------
    constexpr int NUM_STEPS        = 1200;  // Increased for longer simulation
    const std::string output_path  = "sim_output.json";

    sim::SimExporter exporter(output_path, maze);

    for (int t = 0; t < NUM_STEPS; ++t) {
        exporter.record(graph, t);
        
        if ((t + 1) % 10 == 0 || t < 5) {
            std::cout << "\n=== Step " << (t + 1) << " ===\n";
            std::cout << "Nodes: " << graph.nodes.size() << "\n";
        }
        
        sim::step(graph, nn, target, maze);
    }
    // Record final state
    exporter.record(graph, NUM_STEPS);
    exporter.finish();

    std::cout << "Done. Output written to: " 
              << std::filesystem::absolute(output_path).string() << "\n";
    return 0;
}