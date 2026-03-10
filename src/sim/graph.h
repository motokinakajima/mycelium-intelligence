#pragma once

#include "node_nn/nn.h"
#include <array>
#include <vector>

namespace sim {

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

// Input computation
constexpr float WALL_PRESSURE_COEFF = 10.0f;  // coefficient for 1/r^2 wall repulsion
constexpr float CROWD_RADIUS        = 5.0f;   // radius for crowdedness computation

// Output thresholds & multipliers
constexpr float THRESHOLD_APOPTOSIS = 0.8f;   // tanh > 0.8 => node dies
constexpr float THRESHOLD_DEAD_EDGE = 0.1f;   // edges below this weight are pruned
constexpr float PRUNE_EXPONENT      = 3.0f;   // sharpness of prune cone
constexpr float GROW_MULTIPLIER     = 1.5f;   // grow vector length multiplier

// Snap logic
constexpr float SNAP_ANGLE_COS  = 0.866f;  // cos(30 deg) for angle-snap reinforcement
constexpr float THRESHOLD_SPROUT = 0.5f;   // minimum grow strength to sprout new node
constexpr float SNAP_RADIUS     = 3.0f;    // anastomosis snap radius
constexpr float INITIAL_WEIGHT  = 0.5f;    // initial weight for new edges

// Shift logic
constexpr float SHIFT_RATE = 0.5f;  // max node movement per step
constexpr float WALL_AVOIDANCE_STRENGTH = 0.3f;  // heuristic bias away from walls

// Maximum input clamp value (applied before feeding into NN)
constexpr float INPUT_CLAMP = 5.0f;

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
