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
#include "graph.h"    // sim::Graph, sim::step, sim::cleanup_dead
#include "maze.h"     // sim::generate_maze, sim::Maze
#include "export.h"   // sim::SimExporter
#include "config.h"   // sim::load_config

#include <iostream>
#include <filesystem>

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
    constexpr unsigned MAZE_SEED = 42u;

    sim::Maze maze = sim::generate_maze(MAZE_COLS, MAZE_ROWS, MAZE_SEED);
    std::cout << "Maze: " << maze.width << " x " << maze.height << " cells\n";

    // Entry: enclosed start point at cell (1, 1) -> center (1.5, 1.5)
    // Exit:  enclosed end point at cell (W-2, H-2) -> center (W-1.5, H-1.5)
    const sim::Vec2 start  = { 1.5f, 1.5f };
    const sim::Vec2 target = { static_cast<float>(maze.width)  - 1.5f,
                               static_cast<float>(maze.height) - 1.5f };

    std::cout << "Start:  (" << start.x  << ", " << start.y  << ")\n";
    std::cout << "Target: (" << target.x << ", " << target.y << ")\n";

    // ---- Initial graph: single seed node at entry --------------------
    sim::Graph graph;
    {
        sim::Node seed;
        seed.pos     = start;
        seed.is_dead = false;
        graph.nodes.push_back(seed);
    }

    // ---- Neural network: random initialisation (no training data) ----
    node_nn::NeuralNetwork nn;   // constructor calls randomize()

    // For this dummy experiment, bias the output layer so that:
    //   - O[6] (Apoptosis) starts close to -1  -> nodes won't self-destruct
    //   - O[0..1] (Grow)   biased positive     -> encourage outward growth
    // In a real run these biases are learned from training data.
    nn.b2[6] = -5.0f;   // tanh(-5) ≈ -0.9999 -> apoptosis strongly suppressed
    nn.b2[0] =  2.0f;   // nudge Grow X positive (tanh(2) ≈ 0.964)
    nn.b2[1] =  2.0f;   // nudge Grow Y positive
    
    std::cout << "NeuralNetwork: random weights (INPUT=" << node_nn::INPUT_SIZE
              << ", HIDDEN=" << node_nn::HIDDEN_SIZE
              << ", OUTPUT=" << node_nn::OUTPUT_SIZE << ")\n";

    // ---- Run simulation ----------------------------------------------
    constexpr int NUM_STEPS        = 300;  // Increased for longer simulation
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