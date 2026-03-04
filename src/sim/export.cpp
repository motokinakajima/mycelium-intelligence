#include "export.h"
#include <iomanip>
#include <iostream>

namespace sim {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void write_maze_object(std::ostream& os, const Maze& maze) {
    os << "{\"width\":" << maze.width
       << ",\"height\":" << maze.height
       << ",\"grid\":[";

    for (int row = 0; row < maze.height; ++row) {
        if (row > 0) os << ',';
        os << '[';
        for (int col = 0; col < maze.width; ++col) {
            if (col > 0) os << ',';
            os << maze.grid[row][col];
        }
        os << ']';
    }
    os << "]}";
}

static void write_graph_object(std::ostream& os,
                                const Graph&  graph,
                                int           step_number)
{
    os << std::fixed << std::setprecision(4);
    os << "{\"step\":" << step_number << ",\"nodes\":[";

    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        if (i > 0) os << ',';
        const Node& n = graph.nodes[i];
        os << "{\"id\":" << i
           << ",\"x\":" << n.pos.x
           << ",\"y\":" << n.pos.y
           << ",\"edges\":[";

        for (int j = 0; j < static_cast<int>(n.edges.size()); ++j) {
            if (j > 0) os << ',';
            const Edge& e = n.edges[j];
            os << "{\"to\":" << e.target_node_idx
               << ",\"w\":"  << e.weight << '}';
        }
        os << "]}";
    }
    os << "]}";
}

// ---------------------------------------------------------------------------
// write_graph_json
// ---------------------------------------------------------------------------

bool write_graph_json(const std::string& filepath,
                      const Graph&        graph,
                      int                 step_number)
{
    std::ofstream ofs(filepath);
    if (!ofs) {
        std::cerr << "[export] Cannot open " << filepath << " for writing\n";
        return false;
    }
    write_graph_object(ofs, graph, step_number);
    return ofs.good();
}

// ---------------------------------------------------------------------------
// write_maze_json
// ---------------------------------------------------------------------------

bool write_maze_json(const std::string& filepath, const Maze& maze) {
    std::ofstream ofs(filepath);
    if (!ofs) {
        std::cerr << "[export] Cannot open " << filepath << " for writing\n";
        return false;
    }
    write_maze_object(ofs, maze);
    return ofs.good();
}

// ---------------------------------------------------------------------------
// SimExporter
// ---------------------------------------------------------------------------

SimExporter::SimExporter(const std::string& filepath, const Maze& maze)
    : first_step_(true), finished_(false)
{
    ofs_.open(filepath);
    if (!ofs_) {
        std::cerr << "[export] Cannot open " << filepath << " for writing\n";
        return;
    }
    ofs_ << "{\"maze\":";
    write_maze_object(ofs_, maze);
    ofs_ << ",\"steps\":[";
}

SimExporter::~SimExporter() {
    if (!finished_) finish();
}

bool SimExporter::record(const Graph& graph, int step_number) {
    if (!ofs_.is_open() || finished_) return false;
    if (!first_step_) ofs_ << ',';
    first_step_ = false;
    write_graph_object(ofs_, graph, step_number);
    return ofs_.good();
}

void SimExporter::finish() {
    if (finished_) return;
    finished_ = true;
    if (ofs_.is_open()) {
        ofs_ << "]}";
        ofs_.close();
    }
}

} // namespace sim
