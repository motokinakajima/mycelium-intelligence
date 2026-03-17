#include "node_nn/nn.h"
#include "node_nn/utils/io.h"
#include "graph.h"
#include "maze.h"
#include "config.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string find_config_path(const std::string& cli_path) {
    if (!cli_path.empty()) return cli_path;

    const std::vector<std::string> candidates = {
        "hyperparameters.txt",
        "../hyperparameters.txt"
    };

    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    return "";
}

bool load_model_with_fallback(node_nn::NeuralNetwork& nn, std::string& loaded_path) {
    const std::vector<std::string> model_paths = {
        "node_nn_model.nn",
        "../node_nn_model.nn"
    };

    for (const auto& p : model_paths) {
        if (node_nn::load_model(p, nn)) {
            loaded_path = p;
            return true;
        }
    }
    return false;
}

sim::Graph build_initial_graph(const sim::Maze& maze) {
    sim::Graph graph;

    const int start_col = 1;
    const int start_row = 1;
    const int end_col = maze.width - 2;
    const int end_row = maze.height - 2;

    std::vector<std::vector<int>> cell_to_node(
        maze.height,
        std::vector<int>(maze.width, -1));

    for (int row = 0; row < maze.height; ++row) {
        for (int col = 0; col < maze.width; ++col) {
            if (maze.grid[row][col] != 0) continue;

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
        if (row_b < 0 || row_b >= maze.height || col_b < 0 || col_b >= maze.width) return;

        int idx_a = cell_to_node[row_a][col_a];
        int idx_b = cell_to_node[row_b][col_b];
        if (idx_a < 0 || idx_b < 0) return;

        graph.nodes[idx_a].edges.push_back({idx_b, sim::INITIAL_WEIGHT});
        graph.nodes[idx_b].edges.push_back({idx_a, sim::INITIAL_WEIGHT});
    };

    for (int row = 0; row < maze.height; ++row) {
        for (int col = 0; col < maze.width; ++col) {
            if (cell_to_node[row][col] < 0) continue;
            connect_if_open(row, col, row, col + 1);
            connect_if_open(row, col, row + 1, col);
        }
    }

    return graph;
}

std::vector<int> source_indices(const sim::Graph& graph) {
    std::vector<int> ids;
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        if (graph.nodes[i].is_dead) continue;
        if (graph.nodes[i].is_source) ids.push_back(i);
    }
    return ids;
}

bool is_connected_threshold(const sim::Graph& graph, int src, int dst, float min_weight) {
    if (src < 0 || dst < 0 || src >= static_cast<int>(graph.nodes.size()) ||
        dst >= static_cast<int>(graph.nodes.size()))
        return false;

    std::vector<char> visited(static_cast<size_t>(graph.nodes.size()), 0);
    std::queue<int> q;

    visited[src] = 1;
    q.push(src);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        if (u == dst) return true;

        for (const auto& e : graph.nodes[u].edges) {
            if (e.weight < min_weight) continue;
            int v = e.target_node_idx;
            if (v < 0 || v >= static_cast<int>(graph.nodes.size())) continue;
            if (graph.nodes[v].is_dead) continue;
            if (visited[v]) continue;
            visited[v] = 1;
            q.push(v);
        }
    }
    return false;
}

std::string run_trial_rows(
    const node_nn::NeuralNetwork& nn,
    unsigned maze_seed,
    int maze_cols,
    int maze_rows,
    int num_steps) {
    const sim::Maze maze = sim::generate_maze(maze_cols, maze_rows, maze_seed);
    sim::Graph graph = build_initial_graph(maze);
    const sim::Vec2 target = {
        static_cast<float>(maze.width) - 1.5f,
        static_cast<float>(maze.height) - 1.5f
    };

    std::ostringstream oss;
    for (int t = 0; t <= num_steps; ++t) {
        const auto srcs = source_indices(graph);
        bool connected = false;

        if (srcs.size() >= 2) {
            connected = is_connected_threshold(graph, srcs.front(), srcs.back(), 1.0e-6f);
        }

        oss << maze_seed << ','
            << t << ','
            << (connected ? 1 : 0) << ','
            << graph.nodes.size()
            << '\n';

        if (t < num_steps) {
            sim::step(graph, nn, target, maze);
        }
    }

    return oss.str();
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::unitbuf);

    const std::string cli_config = (argc > 1) ? argv[1] : "";
    const std::string output_csv = (argc > 2) ? argv[2] : "seed_step_metrics.csv";
    const int num_trials = (argc > 3) ? std::max(1, std::stoi(argv[3])) : 512;
    const int num_steps = (argc > 4) ? std::max(1, std::stoi(argv[4])) : 1200;
    const int maze_cols = (argc > 5) ? std::max(2, std::stoi(argv[5])) : 5;
    const int maze_rows = (argc > 6) ? std::max(2, std::stoi(argv[6])) : 5;
    const unsigned seed_start = (argc > 7) ? static_cast<unsigned>(std::stoul(argv[7])) : 0u;
    const bool append_mode = (argc > 8) ? (std::stoi(argv[8]) != 0) : false;
    int num_threads = (argc > 9) ? std::max(1, std::stoi(argv[9])) : 24;
    if (num_threads > num_trials) {
        num_threads = num_trials;
    }

    const std::string config_path = find_config_path(cli_config);
    if (!config_path.empty()) {
        std::cout << "Loading config: " << config_path << "\n";
        if (!sim::load_config(config_path)) {
            std::cout << "Config load failed; using defaults.\n";
            sim::set_default_config();
        }
    } else {
        std::cout << "Config not found; using defaults.\n";
        sim::set_default_config();
    }

    node_nn::NeuralNetwork nn;
    std::string model_path;
    if (!load_model_with_fallback(nn, model_path)) {
        std::cerr << "Error: Could not load trained model node_nn_model.nn\n";
        return 1;
    }
    std::cout << "Loaded model: " << model_path << "\n";

    std::ofstream ofs(
        output_csv,
        append_mode ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!ofs) {
        std::cerr << "Error: cannot write output CSV: " << output_csv << "\n";
        return 1;
    }

    if (!append_mode) {
        ofs << "seed,step,connected,node_count\n";
    }

    std::cout << "Running with " << num_threads << " threads\n";

    std::vector<std::string> trial_rows(static_cast<size_t>(num_trials));
    std::atomic<int> next_trial{0};
    std::atomic<int> completed_trials{0};
    std::mutex cout_mutex;

    auto worker = [&]() {
        while (true) {
            const int trial = next_trial.fetch_add(1);
            if (trial >= num_trials) {
                break;
            }

            const unsigned maze_seed = seed_start + static_cast<unsigned>(trial);
            trial_rows[static_cast<size_t>(trial)] = run_trial_rows(
                nn,
                maze_seed,
                maze_cols,
                maze_rows,
                num_steps);

            const int done = completed_trials.fetch_add(1) + 1;
            if ((done % 25 == 0) || done == num_trials) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Progress: " << done << "/" << num_trials << " seeds\n";
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(worker);
    }

    for (auto& th : workers) {
        th.join();
    }

    for (int trial = 0; trial < num_trials; ++trial) {
        ofs << trial_rows[static_cast<size_t>(trial)];
    }

    std::cout << "Done. Wrote metrics: " << std::filesystem::absolute(output_csv).string() << "\n";
    return 0;
}
