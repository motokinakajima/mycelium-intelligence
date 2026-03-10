#include "graph.h"
#include "maze.h"
#include "node_nn/nn.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

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

// Add weight to an edge between two nodes, creating it if it doesn't exist.
// Prevents duplicate edges by checking first.
static void add_or_strengthen_edge(Graph& graph, int from_idx, int to_idx, float weight_delta) {
    if (from_idx < 0 || to_idx < 0 || 
        from_idx >= static_cast<int>(graph.nodes.size()) ||
        to_idx >= static_cast<int>(graph.nodes.size()))
        return;
    
    Node& from_node = graph.nodes[from_idx];
    
    // Look for existing edge
    for (Edge& e : from_node.edges) {
        if (e.target_node_idx == to_idx) {
            // Edge exists, strengthen it
            e.weight += weight_delta;
            return;
        }
    }
    
    // Edge doesn't exist, create it
    from_node.edges.push_back({to_idx, weight_delta});
}

// Raycast from start to end, return the furthest valid (non-wall) position.
// Always checks the entire path for walls, applying a safety margin.
static Vec2 raycast_to_wall(const Maze& maze, const Vec2& start, const Vec2& end) {
    // WALL_SAFETY_MARGIN and RAYCAST_STEP are now loaded from config
    
    // If start is in a wall, can't move
    if (is_wall(maze, start.x, start.y))
        return start;
    
    // Calculate direction and total distance
    float dx = end.x - start.x;
    float dy = end.y - start.y;
    float total_dist = std::sqrt(dx * dx + dy * dy);
    
    if (total_dist < 1.0e-6f)
        return start;
    
    // Normalized direction vector
    float dir_x = dx / total_dist;
    float dir_y = dy / total_dist;
    
    // March along the ray
    float dist = 0.0f;
    float last_safe_dist = 0.0f;
    
    while (dist <= total_dist) {
        Vec2 pos = {start.x + dir_x * dist, start.y + dir_y * dist};
        
        if (is_wall(maze, pos.x, pos.y)) {
            // Hit wall - return position pulled back by safety margin
            float safe_dist = std::max(0.0f, last_safe_dist - WALL_SAFETY_MARGIN);
            return {start.x + dir_x * safe_dist, start.y + dir_y * safe_dist};
        }
        
        last_safe_dist = dist;
        dist += RAYCAST_STEP;
    }
    
    // Entire path is clear - still apply safety margin from endpoint if needed
    // Check one more time at exact endpoint
    if (is_wall(maze, end.x, end.y)) {
        float safe_dist = std::max(0.0f, last_safe_dist - WALL_SAFETY_MARGIN);
        return {start.x + dir_x * safe_dist, start.y + dir_y * safe_dist};
    }
    
    return end;
}

// Check if a line segment from p1 to p2 crosses through any wall cells.
// Returns true if the edge would pass through a wall.
static bool edge_crosses_wall(const Maze& maze, const Vec2& p1, const Vec2& p2) {
    // EDGE_CHECK_STEP is now loaded from config
    
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    
    if (dist < 1.0e-6f)
        return false;
    
    float dir_x = dx / dist;
    float dir_y = dy / dist;
    
    // Sample points along the edge
    for (float t = 0.0f; t <= dist; t += EDGE_CHECK_STEP) {
        Vec2 pos = {p1.x + dir_x * t, p1.y + dir_y * t};
        if (is_wall(maze, pos.x, pos.y))
            return true;
    }
    
    // Final check at endpoint
    if (is_wall(maze, p2.x, p2.y))
        return true;
    
    return false;
}

// Nearest vector from pos to the closest point ON THE SURFACE of any wall cell.
// Using rectangle boundary (not cell centre) gives physically accurate distances
// and prevents 1/r^2 from blowing up when a node sits just outside a wall face.
static Vec2 nearest_wall_vec(const Maze& maze, const Vec2& pos) {
    int cx = static_cast<int>(pos.x);
    int cy = static_cast<int>(pos.y);

    float best_dist2 = std::numeric_limits<float>::max();
    Vec2  best_vec   = {0.0f, 0.0f};

    const int R = 6;  // search radius in cells
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int gx = cx + dx;
            int gy = cy + dy;

            bool  wall;
            float cell_x0, cell_y0, cell_x1, cell_y1;
            if (gx < 0 || gy < 0 || gx >= maze.width || gy >= maze.height) {
                // Treat out-of-bounds as wall; clamp cell to grid boundary
                int bx  = std::max(0, std::min(gx, maze.width  - 1));
                int by  = std::max(0, std::min(gy, maze.height - 1));
                cell_x0 = static_cast<float>(bx);
                cell_y0 = static_cast<float>(by);
                cell_x1 = cell_x0 + 1.0f;
                cell_y1 = cell_y0 + 1.0f;
                wall    = true;
            } else {
                wall    = (maze.grid[gy][gx] == 1);
                cell_x0 = static_cast<float>(gx);
                cell_y0 = static_cast<float>(gy);
                cell_x1 = cell_x0 + 1.0f;
                cell_y1 = cell_y0 + 1.0f;
            }

            if (!wall) continue;

            // Nearest point on the wall cell's AABB to pos
            float nx = clamp(pos.x, cell_x0, cell_x1);
            float ny = clamp(pos.y, cell_y0, cell_y1);
            float d2 = (pos.x - nx) * (pos.x - nx) + (pos.y - ny) * (pos.y - ny);

            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_vec   = {nx - pos.x, ny - pos.y};  // node -> wall surface
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
    // v_wall points from node TO nearest wall surface (boundary, not cell centre).
    // Pressure = -v_hat * COEFF/r^2 (points AWAY from wall).
    // R_MIN prevents divergence and should match the safety margin used in raycasting.
    // Magnitude is also clamped to prevent exceeding INPUT_CLAMP before normalization.
    // R_MIN is now loaded from config
    Vec2  v_wall = nearest_wall_vec(maze, node.pos);
    float r      = std::max(vec2_length(v_wall), R_MIN);
    {
        float inv_r = 1.0f / r;
        float mag   = WALL_PRESSURE_COEFF * inv_r * inv_r;  // COEFF / r^2
        mag = std::min(mag, INPUT_CLAMP);  // clamp magnitude to prevent overflow
        inp[0] = -v_wall.x * inv_r * mag;  // away from wall
        inp[1] = -v_wall.y * inv_r * mag;
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

    if (DEBUG_GROW && grow_len > 0.01f) {
        std::cout << "[DEBUG] Node " << node_idx << " at (" << node.pos.x << ", " << node.pos.y 
                  << ") - Grow output: (" << output[0] << ", " << output[1] 
                  << ") -> len=" << grow_len << " (threshold=" << THRESHOLD_SPROUT << ")\n";
    }

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
                if (DEBUG_GROW) {
                    std::cout << "[DEBUG]   -> Angle-snapped to edge " << e.target_node_idx 
                              << " (cos=" << cos_theta << ", weight+=" << (grow_len * cos_theta) << ")\n";
                }
            }
        }

        // C2. Spatial-snap / sprout if no angle match and grow is strong enough
        if (!snapped && grow_len > THRESHOLD_SPROUT) {
            if (DEBUG_GROW) {
                std::cout << "[DEBUG]   -> Attempting sprout (no angle snap, grow_len > threshold)\n";
            }
            
            Vec2 P_target = {node.pos.x + V_grow.x,
                            node.pos.y + V_grow.y};

            // Raycast to find the furthest valid position (stops at walls)
            Vec2 P_new = raycast_to_wall(maze, node.pos, P_target);
            
            if (DEBUG_GROW) {
                std::cout << "[DEBUG]   -> Raycast: target (" << P_target.x << ", " << P_target.y 
                          << ") -> safe (" << P_new.x << ", " << P_new.y << ")\n";
            }
            
            // Only proceed if we moved a meaningful distance
            float dx_moved = P_new.x - node.pos.x;
            float dy_moved = P_new.y - node.pos.y;
            float dist_moved = std::sqrt(dx_moved * dx_moved + dy_moved * dy_moved);
            
            if (DEBUG_GROW) {
                std::cout << "[DEBUG]   -> Dist moved: " << dist_moved << " (min threshold: " << MIN_SPROUT_DISTANCE << ")\n";
            }
            
            if (dist_moved >= MIN_SPROUT_DISTANCE) {  // >= to include boundary value
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
                    // Anastomosis: connect to existing nearby node (bidirectional)
                    // But first check if the edge would cross through walls
                    const Vec2& target_pos = graph.nodes[nearest_idx].pos;
                    if (!edge_crosses_wall(maze, node.pos, target_pos)) {
                        // Edge is valid - strengthen or create it
                        add_or_strengthen_edge(graph, node_idx, nearest_idx, INITIAL_WEIGHT);
                        add_or_strengthen_edge(graph, nearest_idx, node_idx, INITIAL_WEIGHT);
                        if (DEBUG_GROW) {
                            std::cout << "[DEBUG]   -> Anastomosis to node " << nearest_idx << "\n";
                        }
                    } else if (DEBUG_GROW) {
                        std::cout << "[DEBUG]   -> Anastomosis blocked (edge crosses wall)\n";
                    }
                } else {
                    // Sprout: create a brand-new node (bidirectional connection)
                    Node new_node;
                    new_node.pos     = P_new;
                    new_node.is_dead = false;
                    graph.nodes.push_back(new_node);
                    int new_idx = static_cast<int>(graph.nodes.size()) - 1;
                    // Note: node reference is now invalid (vector may have reallocated)
                    // Create bidirectional edges (new node has no edges yet, so just add)
                    add_or_strengthen_edge(graph, node_idx, new_idx, INITIAL_WEIGHT);
                    add_or_strengthen_edge(graph, new_idx, node_idx, INITIAL_WEIGHT);
                    if (DEBUG_GROW) {
                        std::cout << "[DEBUG]   -> NEW NODE created at (" << P_new.x << ", " << P_new.y 
                                  << "), idx=" << new_idx << "\n";
                    }
                }
            } else if (DEBUG_GROW) {
                std::cout << "[DEBUG]   -> Sprout blocked (dist_moved too small)\n";
            }
        } else if (DEBUG_GROW && !snapped) {
            std::cout << "[DEBUG]   -> No sprout (grow_len " << grow_len << " <= threshold " << THRESHOLD_SPROUT << ")\n";
        }
    }

    // ---- D. Shift ---------------------------------------------------------
    // Wall-slide: try full movement first, then X-only / Y-only fallbacks so
    // a node moving diagonally toward a wall slides along it rather than stopping.
    // Use raycast to prevent nodes from moving into or too close to walls.
    // Add heuristic: bias movement away from walls using wall pressure.
    // CRITICAL: validate that the move won't cause existing edges to cross walls.
    // ANTI-STUCK: if too close to a wall, force movement away from it.
    
    // Get wall pressure for this node
    Vec2 v_wall = nearest_wall_vec(maze, graph.nodes[node_idx].pos);
    float r = vec2_length(v_wall);
    
    // Check if stuck to wall (too close)
    if (r < WALL_STUCK_THRESHOLD) {
        // Force movement away from wall, ignoring NN output
        Vec2 escape_dir = vec2_normalize({-v_wall.x, -v_wall.y});
        Vec2 escape_pos = {
            graph.nodes[node_idx].pos.x + escape_dir.x * WALL_UNSTUCK_FORCE,
            graph.nodes[node_idx].pos.y + escape_dir.y * WALL_UNSTUCK_FORCE
        };
        
        // Validate escape position (must not be in wall)
        if (!is_wall(maze, escape_pos.x, escape_pos.y)) {
            // Also check edges won't cross walls
            bool edges_ok = true;
            const Node& n = graph.nodes[node_idx];
            for (const Edge& e : n.edges) {
                if (e.weight < 0.0f) continue;
                if (e.target_node_idx < 0 ||
                    e.target_node_idx >= static_cast<int>(graph.nodes.size()))
                    continue;
                const Vec2& target_pos = graph.nodes[e.target_node_idx].pos;
                if (edge_crosses_wall(maze, escape_pos, target_pos)) {
                    edges_ok = false;
                    break;
                }
            }
            
            if (edges_ok) {
                graph.nodes[node_idx].pos = escape_pos;
                return;  // Skip normal shift logic
            }
        }
        // If escape failed, continue to normal shift logic (might help incrementally)
    }
    
    // Normal shift logic continues below
    r = std::max(r, 0.3f);
    Vec2 wall_push = {0.0f, 0.0f};
    {
        float inv_r = 1.0f / r;
        float mag = WALL_PRESSURE_COEFF * inv_r * inv_r;
        mag = std::min(mag, INPUT_CLAMP);
        wall_push.x = -v_wall.x * inv_r * mag;
        wall_push.y = -v_wall.y * inv_r * mag;
    }
    
    // Combine NN output with wall-avoidance heuristic
    Vec2 V_shift = {
        output[4] * SHIFT_RATE + wall_push.x * WALL_AVOIDANCE_STRENGTH,
        output[5] * SHIFT_RATE + wall_push.y * WALL_AVOIDANCE_STRENGTH
    };
    const Vec2& cur = graph.nodes[node_idx].pos;  // re-fetch (may have reallocated)

    // Helper lambda: check if moving to new_pos would cause any edge to cross a wall
    auto would_edges_cross_wall = [&](const Vec2& new_pos) -> bool {
        const Node& n = graph.nodes[node_idx];
        for (const Edge& e : n.edges) {
            if (e.weight < 0.0f) continue;  // skip dead edges
            if (e.target_node_idx < 0 ||
                e.target_node_idx >= static_cast<int>(graph.nodes.size()))
                continue;
            const Vec2& target_pos = graph.nodes[e.target_node_idx].pos;
            if (edge_crosses_wall(maze, new_pos, target_pos))
                return true;
        }
        return false;
    };

    Vec2 target_pos = {cur.x + V_shift.x, cur.y + V_shift.y};
    Vec2 safe_pos = raycast_to_wall(maze, cur, target_pos);
    
    // If raycast found a valid position different from current, validate edges
    float dx_move = safe_pos.x - cur.x;
    float dy_move = safe_pos.y - cur.y;
    if (dx_move * dx_move + dy_move * dy_move > 1.0e-6f) {
        // Only move if no edges would cross walls
        if (!would_edges_cross_wall(safe_pos)) {
            graph.nodes[node_idx].pos = safe_pos;
        }
        // else: movement blocked by edge topology, stay put
    } else {
        // Try sliding along walls if direct movement failed
        Vec2 slide_x = {cur.x + V_shift.x, cur.y};
        Vec2 slide_y = {cur.x, cur.y + V_shift.y};
        
        Vec2 safe_x = raycast_to_wall(maze, cur, slide_x);
        Vec2 safe_y = raycast_to_wall(maze, cur, slide_y);
        
        float dx_x = safe_x.x - cur.x;
        float dy_x = safe_x.y - cur.y;
        float dx_y = safe_y.x - cur.x;
        float dy_y = safe_y.y - cur.y;
        
        float dist_x = dx_x * dx_x + dy_x * dy_x;
        float dist_y = dx_y * dx_y + dy_y * dy_y;
        
        // Choose the slide direction that moves furthest AND doesn't break edges
        bool can_slide_x = dist_x > 1.0e-6f && !would_edges_cross_wall(safe_x);
        bool can_slide_y = dist_y > 1.0e-6f && !would_edges_cross_wall(safe_y);
        
        if (can_slide_x && (!can_slide_y || dist_x >= dist_y)) {
            graph.nodes[node_idx].pos = safe_x;
        } else if (can_slide_y) {
            graph.nodes[node_idx].pos = safe_y;
        }
        // else: fully blocked, don't move
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
    
    // Also remove edges that cross walls (cleanup corrupted topology)
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        Node& node = graph.nodes[i];
        node.edges.erase(
            std::remove_if(node.edges.begin(), node.edges.end(),
                           [&](const Edge& e) {
                               if (e.target_node_idx < 0 ||
                                   e.target_node_idx >= static_cast<int>(graph.nodes.size()))
                                   return true;  // invalid index
                               // Check if this edge crosses a wall (should never happen, but clean up if it does)
                               // Skip check in cleanup to avoid performance hit - rely on prevention instead
                               return false;
                           }),
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
