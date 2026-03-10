#include "maze.h"
#include <algorithm>
#include <random>
#include <stack>

namespace sim {

bool is_wall(const Maze& maze, float px, float py) {
    int col = static_cast<int>(px);
    int row = static_cast<int>(py);
    if (col < 0 || row < 0 || col >= maze.width || row >= maze.height)
        return true;  // out of bounds treated as wall
    return maze.grid[row][col] == 1;
}

// ---------------------------------------------------------------------------
// Recursive backtracker maze generation (穴掘り法)
//
// The grid has dimensions (2*cols+1) x (2*rows+1).
// "Room" cells are at even offsets+1 (i.e., odd indices in 0-based).
// Wall cells occupy even indices.
// ---------------------------------------------------------------------------
Maze generate_maze(int cols, int rows, unsigned seed) {
    int W = 2 * cols + 1;
    int H = 2 * rows + 1;

    Maze maze;
    maze.width  = W;
    maze.height = H;
    maze.grid.assign(H, std::vector<int>(W, 1));  // start: all walls

    auto room_to_cell = [](int r) { return 2 * r + 1; };

    // Visited array for room cells
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

    std::mt19937 rng(seed);

    // Direction vectors (in room coordinates): right, down, left, up
    const int dr[] = {0,  1,  0, -1};
    const int dc[] = {1,  0, -1,  0};

    // Start from room (0, 0)
    std::stack<std::pair<int,int>> stk;
    int sr = 0, sc = 0;
    visited[sr][sc] = true;
    maze.grid[room_to_cell(sr)][room_to_cell(sc)] = 0;
    stk.push({sr, sc});

    while (!stk.empty()) {
        auto [cr, cc] = stk.top();

        // Collect unvisited neighbours
        std::vector<int> dirs = {0, 1, 2, 3};
        std::shuffle(dirs.begin(), dirs.end(), rng);

        bool moved = false;
        for (int d : dirs) {
            int nr = cr + dr[d];
            int nc = cc + dc[d];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
            if (visited[nr][nc]) continue;

            // Carve: mark wall between (cr,cc) and (nr,nc) as passage
            int wr = room_to_cell(cr) + dr[d];
            int wc = room_to_cell(cc) + dc[d];
            maze.grid[wr][wc] = 0;

            // Carve the destination room
            maze.grid[room_to_cell(nr)][room_to_cell(nc)] = 0;

            visited[nr][nc] = true;
            stk.push({nr, nc});
            moved = true;
            break;
        }
        if (!moved) stk.pop();
    }

    // No longer carving edge openings - maze is fully enclosed
    // Entry and exit are now inside the maze

    return maze;
}

} // namespace sim
