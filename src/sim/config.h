#pragma once

#include <string>

namespace sim {

// ---------------------------------------------------------------------------
// Hyperparameters (loaded from external file)
// ---------------------------------------------------------------------------

// Input computation
extern float WALL_PRESSURE_COEFF;
extern float CROWD_RADIUS;
extern float R_MIN;

// Output thresholds & multipliers
extern float THRESHOLD_APOPTOSIS;
extern float THRESHOLD_DEAD_EDGE;
extern float PRUNE_EXPONENT;
extern float GROW_MULTIPLIER;

// Snap logic
extern float SNAP_ANGLE_COS;
extern float THRESHOLD_SPROUT;
extern float SNAP_RADIUS;
extern float INITIAL_WEIGHT;

// Shift logic
extern float SHIFT_RATE;
extern float WALL_AVOIDANCE_STRENGTH;
extern float WALL_STUCK_THRESHOLD;
extern float WALL_UNSTUCK_FORCE;

// Input/Output limits
extern float INPUT_CLAMP;

// Raycast & collision detection
extern float WALL_SAFETY_MARGIN;
extern float RAYCAST_STEP;
extern float EDGE_CHECK_STEP;
extern float MIN_SPROUT_DISTANCE;

// Debug flags
extern bool DEBUG_GROW;
extern bool DEBUG_SHIFT;

// ---------------------------------------------------------------------------
// Configuration loader
// ---------------------------------------------------------------------------

// Load hyperparameters from a file.
// Returns true on success, false on error.
// If the file cannot be opened, defaults are used and the function returns false.
bool load_config(const std::string& filepath);

// Set all hyperparameters to their default values.
void set_default_config();

} // namespace sim
