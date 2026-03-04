#pragma once

#include <vector>

namespace sim {

// ---------------------------------------------------------------------------
// 2-D grid maze
//
// Layout:   grid[row][col], i.e. grid[y][x]
// Values:   0 = passage, 1 = wall
// Coordinates: world position (px, py) maps to cell (floor(px), floor(py)).
// ---------------------------------------------------------------------------

struct Maze {
    int                          width;   // number of columns (x)
    int                          height;  // number of rows    (y)
    std::vector<std::vector<int>> grid;   // grid[y][x]
};

// Returns true if world position (px, py) is inside a wall or out of bounds.
bool is_wall(const Maze& maze, float px, float py);

// Returns the centre of cell (col, row) in world coordinates.
// Cell (0,0) has centre (0.5, 0.5).
inline float cell_cx(int col) { return col + 0.5f; }
inline float cell_cy(int row) { return row + 0.5f; }

// Generate a perfect maze using the recursive-backtracker (穴掘り法).
// Resulting dimensions: (2*cols+1) x (2*rows+1) cells.
//   cols / rows = number of "room" cells in each direction.
// Entry is at the top-left corner; exit at the bottom-right.
Maze generate_maze(int cols, int rows, unsigned seed = 42);

} // namespace sim
