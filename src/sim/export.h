#pragma once

#include "graph.h"
#include "maze.h"
#include <fstream>
#include <string>

namespace sim {

// ---------------------------------------------------------------------------
// JSON export
//
// Outputs are intended to be consumed by an HTML+CSS+JS visualiser.
//
// Graph snapshot format (step_NNNN.json):
// {
//   "step": 42,
//   "nodes": [
//     { "id": 0, "x": 1.5, "y": 3.2, "edges": [{"to": 1, "w": 0.8}] },
//     ...
//   ]
// }
//
// Maze format (maze.json):
// {
//   "width": 11, "height": 11,
//   "grid": [[1,1,...], [1,0,...], ...]   // row-major, 0=passage 1=wall
// }
// ---------------------------------------------------------------------------

// Write the current graph state to a JSON file.
// Returns true on success.
bool write_graph_json(const std::string& filepath,
                      const Graph&        graph,
                      int                 step_number);

// Write the maze layout to a JSON file.
// Returns true on success.
bool write_maze_json(const std::string& filepath, const Maze& maze);

// ---------------------------------------------------------------------------
// Streaming simulation export
//
// Creates a single JSON file containing all steps as an array:
// { "maze": {...}, "steps": [ { "step":0, "nodes":[...] }, ... ] }
//
// Usage:
//   SimExporter ex("output/sim.json", maze);
//   for (int t = 0; t < N; ++t) {
//       step(graph, nn, target, maze);
//       ex.record(graph, t);
//   }
//   ex.finish();
// ---------------------------------------------------------------------------

struct SimExporter {
    explicit SimExporter(const std::string& filepath, const Maze& maze);
    ~SimExporter();

    // Append one step snapshot. Returns false if the file is not open.
    bool record(const Graph& graph, int step_number);

    // Finalise the JSON (close arrays/objects). Must be called once.
    void finish();

private:
    std::ofstream ofs_;
    bool          first_step_;
    bool          finished_;
};

} // namespace sim
