#include "config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace sim {

// ---------------------------------------------------------------------------
// Global hyperparameter variables
// ---------------------------------------------------------------------------

// Input computation
float WALL_PRESSURE_COEFF = 10.0f;
float CROWD_RADIUS        = 5.0f;
float R_MIN               = 0.3f;
bool  TARGET_USE_NEAREST_SOURCE = true;
float TARGET_SOURCE_BLEND = 0.85f;

// Output thresholds & multipliers
float THRESHOLD_APOPTOSIS = 0.8f;
float THRESHOLD_DEAD_EDGE = 0.1f;
float PRUNE_EXPONENT      = 3.0f;
float GROW_MULTIPLIER     = 1.5f;

// Snap logic
float SNAP_ANGLE_COS   = 0.866f;
float THRESHOLD_SPROUT = 0.5f;
float SNAP_RADIUS      = 3.0f;
float INITIAL_WEIGHT   = 0.5f;

// Shift logic
float SHIFT_RATE             = 0.5f;
float WALL_AVOIDANCE_STRENGTH = 0.3f;
float WALL_STUCK_THRESHOLD    = 0.6f;
float WALL_UNSTUCK_FORCE      = 0.8f;

// Input/Output limits
float INPUT_CLAMP = 5.0f;

// Raycast & collision detection
float WALL_SAFETY_MARGIN = 0.05f;
float RAYCAST_STEP       = 0.05f;
float EDGE_CHECK_STEP    = 0.1f;
float MIN_SPROUT_DISTANCE = 0.05f;

// Debug flags
bool DEBUG_GROW  = false;
bool DEBUG_SHIFT = false;

// Energy dynamics
float ENERGY_SOURCE_VALUE    = 100.0f;
float ENERGY_MAINTENANCE_COST = 1.0f;
float ENERGY_MAINTENANCE_PER_WEIGHT = 0.02f;
float ENERGY_DIFFUSION_ALPHA  = 0.05f;
float ENERGY_FLOW_GAIN        = 1.0f;
bool  ENERGY_FLOW_NORMALIZE_BY_DEGREE = true;
float ENERGY_INITIAL          = 20.0f;
bool  ENERGY_USE_AS_IMPORTANCE = true;
float ENERGY_IMPORTANCE_SCALE  = 0.05f;
float ENERGY_DEATH_PATIENCE    = 6.0f;
float ENERGY_COST_EDGE_THICKEN = 0.8f;
float ENERGY_COST_NEW_CONNECTION = 1.5f;
float ENERGY_COST_SPROUT = 8.0f;
float ENERGY_CHILD_INITIAL = 6.0f;
float ENERGY_MIN_CLAMP = -50.0f;
float ENERGY_MAX_CLAMP = 200.0f;

float APOPTOSIS_WARMUP_STEPS   = 20.0f;
float NN_APOPTOSIS_ENERGY_GATE = 0.45f;

float FUSION_DISTANCE            = 0.75f;
float FUSION_MAX_MERGES_PER_STEP = 8.0f;
float FUSION_MIN_RETAIN_RATIO    = 0.90f;

bool ENABLE_BACKBONE_PROTECTION = false;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

static bool parse_line(const std::string& line, std::string& key, float& value) {
    // Skip empty lines and comments
    std::string trimmed = trim(line);
    if (trimmed.empty() || trimmed[0] == '#') return false;
    
    // Find '=' separator
    size_t eq_pos = trimmed.find('=');
    if (eq_pos == std::string::npos) return false;
    
    key = trim(trimmed.substr(0, eq_pos));
    std::string value_str = trim(trimmed.substr(eq_pos + 1));
    
    // Parse float value
    try {
        value = std::stof(value_str);
        return true;
    } catch (...) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Configuration loader
// ---------------------------------------------------------------------------

void set_default_config() {
    WALL_PRESSURE_COEFF = 10.0f;
    CROWD_RADIUS        = 5.0f;
    R_MIN               = 0.3f;
    TARGET_USE_NEAREST_SOURCE = true;
    TARGET_SOURCE_BLEND = 0.85f;
    
    THRESHOLD_APOPTOSIS = 0.8f;
    THRESHOLD_DEAD_EDGE = 0.1f;
    PRUNE_EXPONENT      = 3.0f;
    GROW_MULTIPLIER     = 1.5f;
    
    SNAP_ANGLE_COS   = 0.866f;
    THRESHOLD_SPROUT = 0.5f;
    SNAP_RADIUS      = 3.0f;
    INITIAL_WEIGHT   = 0.5f;
    
    SHIFT_RATE             = 0.5f;
    WALL_AVOIDANCE_STRENGTH = 0.3f;
    WALL_STUCK_THRESHOLD    = 0.6f;
    WALL_UNSTUCK_FORCE      = 0.8f;
    
    INPUT_CLAMP = 5.0f;
    
    WALL_SAFETY_MARGIN = 0.05f;
    RAYCAST_STEP       = 0.05f;
    EDGE_CHECK_STEP    = 0.1f;
    MIN_SPROUT_DISTANCE = 0.05f;

    ENERGY_SOURCE_VALUE     = 100.0f;
    ENERGY_MAINTENANCE_COST = 0.6f;
    ENERGY_MAINTENANCE_PER_WEIGHT = 0.02f;
    ENERGY_DIFFUSION_ALPHA  = 0.15f;
    ENERGY_FLOW_GAIN        = 1.0f;
    ENERGY_FLOW_NORMALIZE_BY_DEGREE = true;
    ENERGY_INITIAL          = 40.0f;
    ENERGY_USE_AS_IMPORTANCE = true;
    ENERGY_IMPORTANCE_SCALE  = 0.05f;
    ENERGY_DEATH_PATIENCE    = 6.0f;
    ENERGY_COST_EDGE_THICKEN = 0.8f;
    ENERGY_COST_NEW_CONNECTION = 1.5f;
    ENERGY_COST_SPROUT = 8.0f;
    ENERGY_CHILD_INITIAL = 6.0f;
    ENERGY_MIN_CLAMP = -50.0f;
    ENERGY_MAX_CLAMP = 200.0f;

    APOPTOSIS_WARMUP_STEPS   = 20.0f;
    NN_APOPTOSIS_ENERGY_GATE = 0.45f;

    FUSION_DISTANCE            = 0.75f;
    FUSION_MAX_MERGES_PER_STEP = 8.0f;
    FUSION_MIN_RETAIN_RATIO    = 0.90f;

    ENABLE_BACKBONE_PROTECTION = false;
}

bool load_config(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs) {
        // Don't print error here - let caller handle it
        set_default_config();
        return false;
    }
    
    std::cout << "[config] Loading hyperparameters from " << filepath << "\n";
    
    std::string line;
    int line_num = 0;
    int loaded = 0;
    
    while (std::getline(ifs, line)) {
        ++line_num;
        std::string key;
        float value;
        
        if (!parse_line(line, key, value)) continue;
        
        // Match key to variable
        bool recognized = true;
        
        if      (key == "WALL_PRESSURE_COEFF")     WALL_PRESSURE_COEFF = value;
        else if (key == "CROWD_RADIUS")            CROWD_RADIUS = value;
        else if (key == "R_MIN")                   R_MIN = value;
        else if (key == "TARGET_USE_NEAREST_SOURCE") TARGET_USE_NEAREST_SOURCE = (value > 0.5f);
        else if (key == "TARGET_SOURCE_BLEND")     TARGET_SOURCE_BLEND = value;
        else if (key == "THRESHOLD_APOPTOSIS")     THRESHOLD_APOPTOSIS = value;
        else if (key == "THRESHOLD_DEAD_EDGE")     THRESHOLD_DEAD_EDGE = value;
        else if (key == "PRUNE_EXPONENT")          PRUNE_EXPONENT = value;
        else if (key == "GROW_MULTIPLIER")         GROW_MULTIPLIER = value;
        else if (key == "SNAP_ANGLE_COS")          SNAP_ANGLE_COS = value;
        else if (key == "THRESHOLD_SPROUT")        THRESHOLD_SPROUT = value;
        else if (key == "SNAP_RADIUS")             SNAP_RADIUS = value;
        else if (key == "INITIAL_WEIGHT")          INITIAL_WEIGHT = value;
        else if (key == "SHIFT_RATE")              SHIFT_RATE = value;
        else if (key == "WALL_AVOIDANCE_STRENGTH") WALL_AVOIDANCE_STRENGTH = value;
        else if (key == "WALL_STUCK_THRESHOLD")    WALL_STUCK_THRESHOLD = value;
        else if (key == "WALL_UNSTUCK_FORCE")      WALL_UNSTUCK_FORCE = value;
        else if (key == "INPUT_CLAMP")             INPUT_CLAMP = value;
        else if (key == "WALL_SAFETY_MARGIN")      WALL_SAFETY_MARGIN = value;
        else if (key == "RAYCAST_STEP")            RAYCAST_STEP = value;
        else if (key == "MIN_SPROUT_DISTANCE")     MIN_SPROUT_DISTANCE = value;
        else if (key == "EDGE_CHECK_STEP")         EDGE_CHECK_STEP = value;
        else if (key == "DEBUG_GROW")              DEBUG_GROW = (value > 0.5f);
        else if (key == "DEBUG_SHIFT")             DEBUG_SHIFT = (value > 0.5f);
        else if (key == "ENERGY_SOURCE_VALUE")     ENERGY_SOURCE_VALUE = value;
        else if (key == "ENERGY_MAINTENANCE_COST") ENERGY_MAINTENANCE_COST = value;
        else if (key == "ENERGY_MAINTENANCE_PER_WEIGHT") ENERGY_MAINTENANCE_PER_WEIGHT = value;
        else if (key == "ENERGY_DIFFUSION_ALPHA")  ENERGY_DIFFUSION_ALPHA = value;
        else if (key == "ENERGY_FLOW_GAIN")        ENERGY_FLOW_GAIN = value;
        else if (key == "ENERGY_FLOW_NORMALIZE_BY_DEGREE") ENERGY_FLOW_NORMALIZE_BY_DEGREE = (value > 0.5f);
        else if (key == "ENERGY_INITIAL")          ENERGY_INITIAL = value;
        else if (key == "ENERGY_USE_AS_IMPORTANCE") ENERGY_USE_AS_IMPORTANCE = (value > 0.5f);
        else if (key == "ENERGY_IMPORTANCE_SCALE")  ENERGY_IMPORTANCE_SCALE = value;
        else if (key == "ENERGY_DEATH_PATIENCE")    ENERGY_DEATH_PATIENCE = value;
        else if (key == "ENERGY_COST_EDGE_THICKEN") ENERGY_COST_EDGE_THICKEN = value;
        else if (key == "ENERGY_COST_NEW_CONNECTION") ENERGY_COST_NEW_CONNECTION = value;
        else if (key == "ENERGY_COST_SPROUT")       ENERGY_COST_SPROUT = value;
        else if (key == "ENERGY_CHILD_INITIAL")     ENERGY_CHILD_INITIAL = value;
        else if (key == "ENERGY_MIN_CLAMP")         ENERGY_MIN_CLAMP = value;
        else if (key == "ENERGY_MAX_CLAMP")         ENERGY_MAX_CLAMP = value;
        else if (key == "APOPTOSIS_WARMUP_STEPS")   APOPTOSIS_WARMUP_STEPS = value;
        else if (key == "NN_APOPTOSIS_ENERGY_GATE") NN_APOPTOSIS_ENERGY_GATE = value;
        else if (key == "FUSION_DISTANCE")          FUSION_DISTANCE = value;
        else if (key == "FUSION_MAX_MERGES_PER_STEP") FUSION_MAX_MERGES_PER_STEP = value;
        else if (key == "FUSION_MIN_RETAIN_RATIO")  FUSION_MIN_RETAIN_RATIO = value;
        else if (key == "ENABLE_BACKBONE_PROTECTION") ENABLE_BACKBONE_PROTECTION = (value > 0.5f);
        else {
            std::cerr << "[config] Warning: unknown parameter '" 
                      << key << "' at line " << line_num << "\n";
            recognized = false;
        }
        
        if (recognized) {
            ++loaded;
            std::cout << "[config]   " << key << " = " << value << "\n";
        }
    }
    
    std::cout << "[config] Loaded " << loaded << " parameters\n";
    std::cout.flush();  // Ensure output is visible immediately
    return true;
}

} // namespace sim
