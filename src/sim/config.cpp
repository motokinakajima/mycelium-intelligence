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
