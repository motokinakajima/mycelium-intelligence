#pragma once

#include "node_nn/nn.h"
#include "config.h"  // Hyperparameters loaded from external file
#include <array>
#include <vector>

namespace sim {

// ---------------------------------------------------------------------------
// Hyperparameters are now defined in config.h/cpp and loaded from file
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct Vec2 {
    float x, y;
};

struct Edge {
    int   target_node_idx;  // index into Graph::nodes
    float weight;           // thickness / strength
};

struct Node {
    Vec2              pos;
    std::vector<Edge> edges;
    bool              is_dead;  // set true -> removed on next cleanup
};

// The whole network graph
struct Graph {
    std::vector<Node> nodes;
};

// Forward declaration so graph.h doesn't depend on maze.h order
struct Maze;

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// Compute the 8-element NN input vector for node at index `node_idx`.
std::array<float, node_nn::INPUT_SIZE> compute_inputs(
    const Graph& graph,
    int          node_idx,
    const Vec2&  target,
    const Maze&  maze);

// Apply the 7-element NN output vector (Vibe) to the graph for node `node_idx`.
// May mark nodes dead, modify edge weights, and add new nodes/edges to `graph`.
void apply_vibe(
    Graph&                                      graph,
    int                                         node_idx,
    const std::array<float, node_nn::OUTPUT_SIZE>& output,
    const Maze&                                 maze);

// Run one full simulation step: for every living node, evaluate the NN and
// apply its output. Newly added nodes are NOT evaluated until the next step.
void step(
    Graph&                      graph,
    const node_nn::NeuralNetwork& nn,
    const Vec2&                 target,
    const Maze&                 maze);

// Remove dead nodes and any edges that reference them.
// Updates all remaining edge target indices to remain valid.
void cleanup_dead(Graph& graph);

} // namespace sim
