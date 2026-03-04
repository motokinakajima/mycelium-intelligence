#include "graph.h"
#include "maze.h"
#include "node_nn/nn.h"

#include <cmath>
#include <algorithm>
#include <limits>

namespace sim {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static float vec2_length(const Vec2& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

static Vec2 vec2_normalize(const Vec2& v) {
    float l = vec2_length(v);
    if (l < 1.0e-6f) return {0.0f, 0.0f};
    return {v.x / l, v.y / l};
}

static float vec2_dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}

static float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Vector from node position to the nearest wall cell centre.
static Vec2 nearest_wall_vec(const Maze& maze, const Vec2& pos) {
    int cx = static_cast<int>(pos.x);
    int cy = static_cast<int>(pos.y);

    float best_dist2 = std::numeric_limits<float>::max();
    Vec2  best_vec   = {0.0f, 0.0f};

    const int R = 8;  // search radius in cells
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int gx = cx + dx;
            int gy = cy + dy;

            bool wall;
            float wx, wy;
            if (gx < 0 || gy < 0 || gx >= maze.width || gy >= maze.height) {
                // Clamp out-of-bounds to nearest boundary cell
                int bx = std::max(0, std::min(gx, maze.width  - 1));
                int by = std::max(0, std::min(gy, maze.height - 1));
                wx   = static_cast<float>(bx) + 0.5f;
                wy   = static_cast<float>(by) + 0.5f;
                wall = true;
            } else {
                wall = (maze.grid[gy][gx] == 1);
                wx   = static_cast<float>(gx) + 0.5f;
                wy   = static_cast<float>(gy) + 0.5f;
            }

            if (!wall) continue;

            float d2 = (pos.x - wx) * (pos.x - wx) + (pos.y - wy) * (pos.y - wy);
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_vec   = {wx - pos.x, wy - pos.y};  // node -> wall
            }
        }
    }
    return best_vec;
}

// ---------------------------------------------------------------------------
// compute_inputs
// ---------------------------------------------------------------------------

std::array<float, node_nn::INPUT_SIZE> compute_inputs(
    const Graph& graph,
    int          node_idx,
    const Vec2&  target,
    const Maze&  maze)
{
    std::array<float, node_nn::INPUT_SIZE> inp{};

    const Node& node = graph.nodes[node_idx];

    // ---- I[0], I[1]: Pressure vector (wall repulsion) --------------------
    Vec2  v_wall = nearest_wall_vec(maze, node.pos);
    float r = vec2_length(v_wall);
    if (r > 1.0e-6f) {
        // P = -(v/r) * (COEFF / r^2)  -> points AWAY from wall
        float mag = WALL_PRESSURE_COEFF / (r * r * r);  // 1/r^3 after normalisation
        inp[0] = -v_wall.x * mag;
        inp[1] = -v_wall.y * mag;
    }

    // ---- I[2], I[3]: Target (goal) vector --------------------------------
    Vec2 v_target = {target.x - node.pos.x, target.y - node.pos.y};
    float td = vec2_length(v_target);
    if (td > 1.0e-6f) {
        inp[2] = v_target.x / td;
        inp[3] = v_target.y / td;
    }

    // ---- I[4], I[5]: Flow COM vector -------------------------------------
    // Weighted sum of normalised edge direction vectors
    Vec2  flow = {0.0f, 0.0f};
    float weight_sum = 0.0f;
    for (const Edge& e : node.edges) {
        if (e.target_node_idx < 0 ||
            e.target_node_idx >= static_cast<int>(graph.nodes.size()))
            continue;
        const Node& tgt = graph.nodes[e.target_node_idx];
        Vec2 ev = {tgt.pos.x - node.pos.x, tgt.pos.y - node.pos.y};
        Vec2 ev_n = vec2_normalize(ev);
        flow.x      += ev_n.x * e.weight;
        flow.y      += ev_n.y * e.weight;
        weight_sum  += e.weight;
    }
    inp[4] = flow.x;
    inp[5] = flow.y;

    // ---- I[6]: Importance (total edge weight) ----------------------------
    inp[6] = weight_sum;

    // ---- I[7]: Crowdedness (neighbours within CROWD_RADIUS) --------------
    float crowd = 0.0f;
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        if (i == node_idx) continue;
        if (graph.nodes[i].is_dead) continue;
        float dx = graph.nodes[i].pos.x - node.pos.x;
        float dy = graph.nodes[i].pos.y - node.pos.y;
        if (dx * dx + dy * dy < CROWD_RADIUS * CROWD_RADIUS)
            crowd += 1.0f;
    }
    inp[7] = crowd;

    // ---- Clamp all inputs ------------------------------------------------
    for (float& v : inp)
        v = clamp(v, -INPUT_CLAMP, INPUT_CLAMP);

    return inp;
}

// ---------------------------------------------------------------------------
// apply_vibe
// ---------------------------------------------------------------------------

void apply_vibe(
    Graph&                                         graph,
    int                                            node_idx,
    const std::array<float, node_nn::OUTPUT_SIZE>& output,
    const Maze&                                    maze)
{
    Node& node = graph.nodes[node_idx];

    // ---- A. Apoptosis -----------------------------------------------------
    if (output[6] > THRESHOLD_APOPTOSIS) {
        node.is_dead = true;
        return;  // skip remaining logic
    }

    // ---- B. Prune ---------------------------------------------------------
    Vec2 V_prune = {output[2], output[3]};
    float prune_len = vec2_length(V_prune);
    if (prune_len > 1.0e-6f) {
        Vec2 V_prune_n = vec2_normalize(V_prune);
        for (Edge& e : node.edges) {
            if (e.target_node_idx < 0 ||
                e.target_node_idx >= static_cast<int>(graph.nodes.size()))
                continue;
            const Node& tgt = graph.nodes[e.target_node_idx];
            Vec2 ev = {tgt.pos.x - node.pos.x, tgt.pos.y - node.pos.y};
            Vec2 ev_n = vec2_normalize(ev);
            float dot_val = vec2_dot(ev_n, V_prune_n);
            if (dot_val > 0.0f) {
                float reduction = prune_len *
                    std::pow(dot_val, PRUNE_EXPONENT);
                e.weight -= reduction;
            }
        }
        // Mark edges below threshold for removal
        for (Edge& e : node.edges)
            if (e.weight < THRESHOLD_DEAD_EDGE)
                e.weight = -1.0f;  // sentinel: removed during cleanup
    }

    // ---- C. Grow & Sprout -------------------------------------------------
    Vec2 V_grow = {output[0] * GROW_MULTIPLIER,
                   output[1] * GROW_MULTIPLIER};
    float grow_len = vec2_length(V_grow);

    bool snapped = false;
    if (grow_len > 1.0e-6f) {
        Vec2 V_grow_n = vec2_normalize(V_grow);

        // C1. Angle-snap: reinforce existing edges close in direction
        for (Edge& e : node.edges) {
            if (e.weight < 0.0f) continue;  // already marked dead
            if (e.target_node_idx < 0 ||
                e.target_node_idx >= static_cast<int>(graph.nodes.size()))
                continue;
            const Node& tgt = graph.nodes[e.target_node_idx];
            Vec2 ev   = {tgt.pos.x - node.pos.x, tgt.pos.y - node.pos.y};
            Vec2 ev_n = vec2_normalize(ev);
            float cos_theta = vec2_dot(ev_n, V_grow_n);
            if (cos_theta > SNAP_ANGLE_COS) {
                e.weight += grow_len * cos_theta;
                snapped = true;
            }
        }

        // C2. Spatial-snap / sprout if no angle match and grow is strong enough
        if (!snapped && grow_len > THRESHOLD_SPROUT) {
            Vec2 P_new = {node.pos.x + V_grow.x,
                          node.pos.y + V_grow.y};

            // Cancel if destination is a wall
            if (!is_wall(maze, P_new.x, P_new.y)) {

                // Check for existing nodes within SNAP_RADIUS
                int nearest_idx = -1;
                float nearest_d2 = SNAP_RADIUS * SNAP_RADIUS;
                for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
                    if (i == node_idx) continue;
                    if (graph.nodes[i].is_dead) continue;
                    float dx = graph.nodes[i].pos.x - P_new.x;
                    float dy = graph.nodes[i].pos.y - P_new.y;
                    float d2 = dx * dx + dy * dy;
                    if (d2 < nearest_d2) {
                        nearest_d2 = d2;
                        nearest_idx = i;
                    }
                }

                if (nearest_idx >= 0) {
                    // Anastomosis: connect to existing nearby node
                    // Check we don't already have this edge
                    bool exists = false;
                    for (const Edge& e : node.edges)
                        if (e.target_node_idx == nearest_idx) { exists = true; break; }
                    if (!exists)
                        node.edges.push_back({nearest_idx, INITIAL_WEIGHT});
                } else {
                    // Sprout: create a brand-new node
                    Node new_node;
                    new_node.pos     = P_new;
                    new_node.is_dead = false;
                    graph.nodes.push_back(new_node);
                    int new_idx = static_cast<int>(graph.nodes.size()) - 1;
                    // Note: node reference is now invalid (vector may have reallocated)
                    graph.nodes[node_idx].edges.push_back({new_idx, INITIAL_WEIGHT});
                }
            }
        }
    }

    // ---- D. Shift ---------------------------------------------------------
    Vec2 V_shift = {output[4] * SHIFT_RATE,
                    output[5] * SHIFT_RATE};
    Vec2 new_pos = {node.pos.x + V_shift.x,
                    node.pos.y + V_shift.y};
    if (!is_wall(maze, new_pos.x, new_pos.y)) {
        // Re-fetch node reference in case vector was reallocated in Sprout
        graph.nodes[node_idx].pos = new_pos;
    }
}

// ---------------------------------------------------------------------------
// step
// ---------------------------------------------------------------------------

void step(
    Graph&                        graph,
    const node_nn::NeuralNetwork& nn,
    const Vec2&                   target,
    const Maze&                   maze)
{
    int n = static_cast<int>(graph.nodes.size());  // fixed: don't process new sprouts
    for (int i = 0; i < n; ++i) {
        if (graph.nodes[i].is_dead) continue;
        auto input  = compute_inputs(graph, i, target, maze);
        std::array<float, node_nn::OUTPUT_SIZE> output{};
        node_nn::forward(nn, input, output);
        apply_vibe(graph, i, output, maze);
    }
    cleanup_dead(graph);
}

// ---------------------------------------------------------------------------
// cleanup_dead
// ---------------------------------------------------------------------------

void cleanup_dead(Graph& graph) {
    // First, remove dead-weight edges (weight == -1 sentinel) from all nodes
    for (Node& node : graph.nodes) {
        node.edges.erase(
            std::remove_if(node.edges.begin(), node.edges.end(),
                           [](const Edge& e) { return e.weight < 0.0f; }),
            node.edges.end());
    }

    // Build index remapping: old_idx -> new_idx (-1 if dead)
    std::vector<int> remap(graph.nodes.size(), -1);
    int new_idx = 0;
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        if (!graph.nodes[i].is_dead)
            remap[i] = new_idx++;
    }

    // Remove dead nodes
    graph.nodes.erase(
        std::remove_if(graph.nodes.begin(), graph.nodes.end(),
                       [](const Node& n) { return n.is_dead; }),
        graph.nodes.end());

    // Update all edge indices; drop edges to dead nodes
    for (Node& node : graph.nodes) {
        std::vector<Edge> valid_edges;
        for (Edge& e : node.edges) {
            if (e.target_node_idx >= 0 &&
                e.target_node_idx < static_cast<int>(remap.size()) &&
                remap[e.target_node_idx] >= 0)
            {
                e.target_node_idx = remap[e.target_node_idx];
                valid_edges.push_back(e);
            }
            // else: edge to dead/invalid node is discarded
        }
        node.edges = std::move(valid_edges);
    }
}

} // namespace sim
