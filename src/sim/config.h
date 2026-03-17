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
extern bool  TARGET_USE_NEAREST_SOURCE;
extern float TARGET_SOURCE_BLEND;

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

// Energy dynamics
extern float ENERGY_SOURCE_VALUE;
extern float ENERGY_MAINTENANCE_COST;
extern float ENERGY_MAINTENANCE_PER_WEIGHT;
extern float ENERGY_DIFFUSION_ALPHA;
extern float ENERGY_FLOW_GAIN;
extern bool  ENERGY_FLOW_NORMALIZE_BY_DEGREE;
extern bool  ENERGY_PULSE_ENABLE;
extern float ENERGY_PULSE_PERIOD_STEPS;
extern float ENERGY_PULSE_LOW_RATIO;
extern float ENERGY_PULSE_SINK_VALUE;
extern float EDGE_WEIGHT_MAX;
extern float ENERGY_INITIAL;
extern bool  ENERGY_USE_AS_IMPORTANCE;
extern float ENERGY_IMPORTANCE_SCALE;
extern float ENERGY_DEATH_PATIENCE;
extern float ENERGY_COST_EDGE_THICKEN;
extern float ENERGY_COST_NEW_CONNECTION;
extern float ENERGY_COST_SPROUT;
extern float ENERGY_CHILD_INITIAL;
extern float ENERGY_MIN_CLAMP;
extern float ENERGY_MAX_CLAMP;

// Apoptosis stabilization
extern float APOPTOSIS_WARMUP_STEPS;
extern float NN_APOPTOSIS_ENERGY_GATE;

// Anastomosis (node merge)
extern float FUSION_DISTANCE;
extern float FUSION_MAX_MERGES_PER_STEP;
extern float FUSION_MIN_RETAIN_RATIO;

// Research/Submission mode switch
extern bool ENABLE_BACKBONE_PROTECTION;

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
