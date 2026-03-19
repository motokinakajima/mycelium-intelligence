# Mycelium Intelligence

A decentralized AI system inspired by fungal mycelium networks, featuring node-level neural networks that collectively navigate mazes through dynamic graph rewriting.

## Overview

Mushroom mycelium is one of the oldest forms of life on Earth, with millennia of evolutionary advantages yet to be explored. Recent discoveries of mycelium intelligence and demonstrations of complex adaptive behaviours have been exciting the scientific world. Fukasawa et al. (2024) found that fungal mycelium can "recognize" the difference in spatial arrangement as mycelium maintained its "X" and "O" spatial structure with growth. This display of spatial awareness compounds upon Money (2021)'s in-depth study of fungal hyphae's extreme sensitivity to their environment, allowing them to detect surface ridges, repair wounds, and other expressions of decentralized biological decision-making.

This project challenges traditional centralized "predictive brain" AI architectures with a decentralized, self-organizing collective of "node-level brains" grounded in *Pleurotus ostreatus* mycelium hyphal activity. Rather than using static, centralized, high-dimensional mapping, we employ dynamic graph rewriting learned from biological growth patterns. Each node runs a small neural network that makes local decisions, and collectively the network exhibits emergent maze-solving behavior.

Similar to biological neural networks, greater activity creates pathways with heavier weights. This is reflected in the hyphal network where frequently used paths strengthen over time through anastomosis (fusion) and pruning of unused branches.

---

## Current Implementation Status

### ✅ Completed Features

- **Node-level Neural Network**: 8-input, 8-hidden, 7-output architecture with tanh activation
- **Dynamic Graph System**: Nodes and weighted edges representing the mycelium network
- **Maze Generation**: Recursive backtracker algorithm for perfect mazes
- **Four Core Behaviors**:
  - **Apoptosis**: Self-destruction of unimportant nodes
  - **Prune**: Directional weakening and removal of edges
  - **Grow & Sprout**: Reinforcement of existing edges or creation of new nodes
  - **Shift**: Node position adjustment for network optimization
- **Wall Collision Handling**: Raycasting and edge validation to prevent wall penetration
- **Anastomosis**: Automatic fusion when growing nodes approach each other
- **Configuration System**: External hyperparameter file (no recompilation needed)
- **Energy Diffusion Model**: Gradient-based local diffusion with per-step outflow cap, maintenance, source injection, clamp, and energy-based death
- **JSON Export**: Complete simulation state export for visualization
- **Debug System**: Detailed logging for growth and movement behaviors
- **Trainer Executable**: Dedicated `train_nn` target for CSV-based supervised training
- **Deterministic Batch Evaluator**: `eval_seed_trials` for multi-seed metrics (`seed,step,connected,node_count`)
- **Parallel Trials**: Batch evaluation with multi-thread execution (default 24 threads)
- **Metrics Analyzer**: `results/analyze_connected_metrics.py` for disappearance events and persistent metrics
- **Hyperparameter Search**: `heuristics/hyperparameter_sweep.py` supports sweep and stochastic hill-climb (optional annealing)
- **Plot Utilities**: `results/visualize_persistent_decay.py` and `results/plot_node_nn_poster_figure.py`

### 🚧 Partially Implemented

- **Biological Grounding**: Training pipeline is implemented, but real biological dataset integration remains ongoing
- **Unstuck Mechanism**: Nodes can escape wall entrapment, but tuning needed

### ⏳ Planned Features

- Biological data collection from *Pleurotus ostreatus* maze experiments
- Training node-level networks on real mycelium behavior
- Multi-target pathfinding
- Network simplification and consolidation algorithms

---

## Architecture

### System Components

```
mycelium-intelligence/
├── src/
│   ├── main.cpp              # Entry point and simulation loop
│   ├── nn/                   # Neural network module
│   │   └── node_nn/
│   │       ├── nn.h/cpp      # Network structure and training
│   │       └── utils/
│   │           └── io.h/cpp  # Model persistence
│   └── sim/                  # Simulation module
│       ├── config.h/cpp      # Hyperparameter management
│       ├── graph.h/cpp       # Core graph logic
│       ├── maze.h/cpp        # Maze generation
│       └── export.h/cpp      # JSON output
└── hyperparameters.txt       # External configuration
```

### Neural Network I/O Specification

**Inputs (8 dimensions)**: Environmental sensors
1. **Pressure Vector X**: Repulsion from nearest wall (X component)
2. **Pressure Vector Y**: Repulsion from nearest wall (Y component)
3. **Nearest Node Vector X**: Direction to nearest alive node (X component, normalized)
4. **Nearest Node Vector Y**: Direction to nearest alive node (Y component, normalized)
5. **Flow COM Vector X**: Net nutrient-flow direction from local diffusion equation (X component)
6. **Flow COM Vector Y**: Net nutrient-flow direction from local diffusion equation (Y component)
7. **Importance**: `node.energy * ENERGY_IMPORTANCE_SCALE` if enabled, otherwise sum of connected edge weights
8. **Crowdedness**: Number of nearby nodes within CROWD_RADIUS

All inputs are clamped to [-INPUT_CLAMP, INPUT_CLAMP] to prevent overflow.

**Outputs (7 dimensions)**: Node's "Vibes" (intentions)
1. **Grow Vector X**: Direction to strengthen or create new connections
2. **Grow Vector Y**: Direction to strengthen or create new connections
3. **Prune Vector X**: Direction to weaken/remove connections
4. **Prune Vector Y**: Direction to weaken/remove connections
5. **Shift Vector X**: Direction the node wants to move
6. **Shift Vector Y**: Direction the node wants to move
7. **Apoptosis**: Self-destruction desire (scalar)

All outputs are in [-1, 1] range due to tanh activation.

### Deterministic Logic: Vibe → Action

For each node, outputs are applied in strict order:

#### A. Apoptosis
```
if output[6] > THRESHOLD_APOPTOSIS:
    node.is_dead = true
    return  // Skip remaining logic
```

#### B. Prune (Edge Weakening)
```
V_prune = (output[2], output[3])
for each edge:
    if dot(edge_direction, V_prune) > 0 and target is not source:
        reduction = length(V_prune) * dot^PRUNE_EXPONENT
        edge.weight -= reduction
        if edge.weight < THRESHOLD_DEAD_EDGE:
            mark for removal
```

#### C. Grow & Sprout
```
V_grow = (output[0], output[1]) * GROW_MULTIPLIER

# C1. Angle-snap: Reinforce existing edges
for each edge:
    cos_theta = dot(edge_direction, V_grow)
    if cos_theta > SNAP_ANGLE_COS:
        applied = min(length(V_grow) * cos_theta, affordable_by_energy)
        edge.weight += applied
        node.energy -= applied * ENERGY_COST_EDGE_THICKEN
        snapped = true

# C2. Spatial-snap / Sprout new node
if not snapped and length(V_grow) > THRESHOLD_SPROUT:
    P_new = raycast_to_wall(node.pos, node.pos + V_grow)
    
    if distance(node.pos, P_new) >= MIN_SPROUT_DISTANCE:
        # Check for nearby nodes (anastomosis)
        nearest = find_node_within(P_new, SNAP_RADIUS)
        
        if nearest exists and not edge_crosses_wall:
            create_bidirectional_edge(node, nearest)  # Anastomosis
        else:
            create_new_node(P_new)
            create_bidirectional_edge(node, new_node)
```

#### D. Shift (Node Movement)
```
# Anti-stuck mechanism
if distance_to_wall < WALL_STUCK_THRESHOLD:
    force_move_away_from_wall(WALL_UNSTUCK_FORCE)
    return

# Normal movement
V_shift = output[4:5] * SHIFT_RATE + wall_pressure_heuristic
P_target = raycast_to_wall(node.pos, node.pos + V_shift)

# Validate: No edges cross walls after movement
if no_edges_would_cross_walls(P_target):
    node.pos = P_target
else:
    try_slide_along_wall()  # Fallback to X-only or Y-only movement
```

### Energy Diffusion Model (Current)

After per-node NN actions, energy is updated synchronously in three stages:

1. **Pairwise gradient diffusion**
    - For each undirected edge `(i, j)`, flux is computed as:
    - `f(i->j) = beta * w_ij * (E_i - E_j)`
    - Positive flux moves energy from `i` to `j`; negative flux means reverse direction.
2. **Per-node outflow cap (alpha-like guarantee)**
    - Raw outgoing flux from each node is capped by:
    - `out_i <= ENERGY_FLOW_GAIN * E_i`
    - This guarantees at least `(1 - ENERGY_FLOW_GAIN) * E_i` remains before maintenance/source/clamp.
3. **Maintenance, source injection, clamp, death**
    - `E_i <- E_i - ENERGY_MAINTENANCE_COST`
    - Source nodes are overwritten to `ENERGY_SOURCE_VALUE`
    - Then clamped to `[ENERGY_MIN_CLAMP, ENERGY_MAX_CLAMP]`
    - Non-source nodes die immediately when `E_i <= 0`
    - Additional gate death is enabled after warmup: `E_i <= NN_APOPTOSIS_ENERGY_GATE`

Post-processing then removes dead/cross-wall edges, performs optional fusion, and enforces bidirectional edge symmetry.

---

## Building and Running

### Prerequisites
- CMake 3.16 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)

### Build Instructions
```bash
cmake -B cmake-build-debug
cmake --build cmake-build-debug
```

### Running the Simulation

Basic usage:
```bash
./cmake-build-debug/mycelium
```

With custom configuration file:
```bash
./cmake-build-debug/mycelium path/to/custom_config.txt
```

The simulation outputs to `sim_output.json` in JSON format, containing:
- Maze layout
- All simulation steps with node positions and edge weights
- Complete network topology evolution

### Batch Evaluation Pipeline

Run evaluator + analyzer in one command:

```bat
scripts\run_eval_and_analyze.bat
```

Defaults used by the batch script:
- `TRIALS=512`
- `STEPS=1200`
- `THREADS=24`
- output CSV: `cmake-build-debug\seed_step_metrics_512.csv`
- analysis output: `results\connected_metrics`

`--persistent-until-step` is auto-derived from:

`ENERGY_PULSE_PERIOD_STEPS * 21.5`

You can override this as the 9th argument:

```bat
scripts\run_eval_and_analyze.bat 512 1200 5 5 0 24 cmake-build-debug seed_step_metrics_512.csv 1100
```

Direct evaluator call:

```bat
cmake-build-debug\eval_seed_trials.exe hyperparameters.txt cmake-build-debug\seed_step_metrics_512.csv 512 1200 5 5 0 0 24
```

Direct analyzer call:

```bat
python results\analyze_connected_metrics.py cmake-build-debug\seed_step_metrics_512.csv --out-dir results\connected_metrics --persistent-until-step 1075
```

### Hyperparameter Search

Use `heuristics/hyperparameter_sweep.py`:

- `--mode sweep`: one-factor + random combinations
- `--mode hill`: stochastic hill-climb (optional annealing)

Example overnight hill-climb:

```bat
python heuristics\hyperparameter_sweep.py --mode hill --build-first --hc-restarts 12 --hc-iters 400 --hc-k 3 --hc-k-fine 1 --hc-fine-after 0.55 --deltas 0.02,0.03 --hc-fine-deltas 0.01 --hc-anneal --hc-temp-start 0.01 --hc-temp-end 0.0005 --hc-patience 80
```

### Plotting Utilities

Persistent decay plot:

```bat
python results\visualize_persistent_decay.py
```

Poster-style conceptual figure:

```bat
python results\plot_node_nn_poster_figure.py --show
```

---

## Configuration

All hyperparameters are defined in `hyperparameters.txt` at the project root. Changes take effect immediately on next run—**no recompilation needed**.

Note: the authoritative current values are those in `hyperparameters.txt` at runtime.

### Hyperparameter Categories

#### Input Computation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `WALL_PRESSURE_COEFF` | 10.0 | Coefficient for 1/r² wall repulsion force |
| `CROWD_RADIUS` | 5.0 | Radius for detecting nearby nodes (crowdedness) |
| `R_MIN` | 0.3 | Minimum distance to wall (prevents divergence) |

#### Output Thresholds & Multipliers
| Parameter | Default | Description |
|-----------|---------|-------------|
| `THRESHOLD_APOPTOSIS` | 0.8 | Nodes self-destruct if output[6] > this |
| `THRESHOLD_DEAD_EDGE` | 0.1 | Edges below this weight are removed |
| `PRUNE_EXPONENT` | 3.0 | Sharpness of pruning cone (higher = more focused) |
| `GROW_MULTIPLIER` | 2.5 | Amplification factor for grow vector |

#### Snap Logic
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SNAP_ANGLE_COS` | 0.866 | cos(30°) - angle threshold for edge reinforcement |
| `THRESHOLD_SPROUT` | 0.3 | Minimum grow strength to create new nodes |
| `SNAP_RADIUS` | 3.0 | Distance for anastomosis (node fusion) |
| `INITIAL_WEIGHT` | 0.5 | Starting weight for newly created edges |

#### Shift Logic
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SHIFT_RATE` | 0.5 | Maximum node movement per step |
| `WALL_AVOIDANCE_STRENGTH` | 0.3 | Heuristic bias away from walls |
| `WALL_STUCK_THRESHOLD` | 0.4 | Distance threshold for "stuck to wall" detection |
| `WALL_UNSTUCK_FORCE` | 1.0 | Force to push nodes away from walls when stuck |

#### Collision Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `WALL_SAFETY_MARGIN` | 0.03 | Distance buffer from walls during movement |
| `RAYCAST_STEP` | 0.05 | Step size for raycasting collision detection |
| `EDGE_CHECK_STEP` | 0.1 | Step size for edge-wall crossing validation |
| `MIN_SPROUT_DISTANCE` | 0.03 | Minimum movement to allow sprouting new node |

#### Debug Flags
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEBUG_GROW` | 0 | Enable detailed logging for growth behavior (1 = on) |
| `DEBUG_SHIFT` | 0 | Enable detailed logging for movement behavior (1 = on) |

#### Energy Dynamics
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENERGY_SOURCE_VALUE` | 100.0 | Energy value forced on source nodes each step |
| `ENERGY_MAINTENANCE_COST` | 0.6 | Per-step energy cost per alive node |
| `ENERGY_DIFFUSION_ALPHA` | 0.15 | Gradient diffusion coefficient `beta` in `f(i->j) = beta * w_ij * (E_i - E_j)` |
| `ENERGY_FLOW_GAIN` | 1.0 | Per-step maximum outgoing fraction of node energy (outflow cap ratio) |
| `ENERGY_FLOW_NORMALIZE_BY_DEGREE` | 1 | Reserved compatibility flag (currently not used in active diffusion path) |
| `ENERGY_INITIAL` | 40.0 | Initial energy for non-source nodes |
| `ENERGY_USE_AS_IMPORTANCE` | 1 | Use energy (scaled) for input[6] instead of weight-sum |
| `ENERGY_IMPORTANCE_SCALE` | 0.05 | Scale factor when energy is used for input[6] |
| `ENERGY_COST_EDGE_THICKEN` | 0.8 | Energy cost per unit edge-thickening |
| `ENERGY_COST_NEW_CONNECTION` | 1.5 | Energy cost for new connection (anastomosis/sprout link) |
| `ENERGY_COST_SPROUT` | 8.0 | Base energy cost to create a new node |
| `ENERGY_CHILD_INITIAL` | 6.0 | Initial energy assigned to a sprouted child node |
| `ENERGY_MIN_CLAMP` | -50.0 | Lower bound after update (before death check) |
| `ENERGY_MAX_CLAMP` | 200.0 | Upper bound after update |

#### Apoptosis / Fusion Stabilization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `APOPTOSIS_WARMUP_STEPS` | 20 | Delay before gate-based apoptosis starts |
| `NN_APOPTOSIS_ENERGY_GATE` | 0.45 | Additional death threshold after warmup |
| `FUSION_DISTANCE` | 0.75 | Max distance for merge candidates |
| `FUSION_MAX_MERGES_PER_STEP` | 8 | Merge attempts per step |
| `FUSION_MIN_RETAIN_RATIO` | 0.90 | Minimum retained incident-weight ratio to allow merge |

---

## Known Issues and Limitations

### Current Challenges

1. **Wall Entrapment**: Nodes sometimes get trapped near walls despite anti-stuck mechanism
   - **Impact**: Limits network exploration
   - **Workaround**: Adjust `WALL_STUCK_THRESHOLD` and `WALL_UNSTUCK_FORCE`
   - **Future Fix**: Implement attraction to open space when stuck

2. **Anastomosis Overuse**: Nodes prefer connecting to existing nodes over creating new ones
   - **Impact**: Network may plateau at 4-6 nodes in complex mazes
   - **Workaround**: Reduce `SNAP_RADIUS` or increase `THRESHOLD_SPROUT`
   - **Future Fix**: Add penalty for redundant connections

3. **No Biological Training Dataset Yet**: Pipeline exists, but model quality still depends on synthetic/heuristic data
   - **Impact**: Behavior is not biomimetic
   - **Solution**: Collect real *Pleurotus ostreatus* maze data and train

4. **Performance**: Debug logging significantly slows simulation
   - **Workaround**: Set `DEBUG_GROW = 0` and `DEBUG_SHIFT = 0` for production runs

5. **Boundary Value Sensitivity**: Small changes in thresholds have large effects
   - **Impact**: Difficult to tune parameters
   - **Solution**: Use adaptive thresholds based on local conditions

### Design Limitations

- **2D Only**: Current implementation is limited to planar mazes
- **Single Target**: Only one goal position supported
- **No Long-Term Memory**: Node decisions remain myopic and per-step local
- **Static Maze**: Walls don't change during simulation

---

## Future Improvements

### High Priority
1. **Biological Data Collection Pipeline**
   - Time-lapse imaging of *Pleurotus ostreatus* in mazes
   - Automated node/edge extraction from images
   - Training dataset generation

2. **Visualization Tool**
   - Web-based playback of `sim_output.json`
   - Real-time simulation viewer
   - Parameter tuning interface

3. **Adaptive Hyperparameters**
   - Context-dependent thresholds
   - Learning rate schedules for NN training

### Medium Priority
- Multi-target pathfinding
- 3D maze support
- Network simplification algorithms (merge redundant nodes)

### Low Priority
- Different maze algorithms (Prim's, Kruskal's, etc.)
- Alternative NN architectures (LSTMs for memory)
- Multi-species competition simulation

---

## Biological Protocol

### Materials
- Still-air box
- 70% ethanol
- Malt extract agar dishes
- *Pleurotus ostreatus* strain (Liquid Fungi)
- 3D-printed maze inserts
- Medical gloves
- Time-lapse camera

### Mycelium Cultivation Protocol
1. **Sterilization**: Clean workspace, still-air box, and containers with 70% ethanol
2. **Inoculation**: Drop 3-4 drops of liquid mycelium spawn on agar center (3 plates)
3. **Maze Preparation**: Sterilize 3D-printed maze inserts in ethanol (15+ minutes)
4. **Transfer**: Cut fully colonized agar pieces with sterilized scalpel, place at maze endpoints
5. **Documentation**: Time-lapse or daily photography until growth stops (5+ days no activity)
6. **Monitoring**: Check colonization every 2 days; record maze dishes daily for 2 weeks

### Computational Protocol
1. **Video Analysis**: Review time-lapse, identify key timestamps
2. **Feature Extraction**: Identify nodes (branch points) and edges (hyphae) from images
3. **Data Annotation**: Record node coordinates and edge weights (thickness-based)
4. **Dataset Creation**: Compile input features (environment) and target outputs (behavior change)
5. **NN Training**: Train 8-8-7 network using Adam optimizer
6. **Validation**: Test trained network in virtual mazes
7. **Iteration**: Refine and retrain based on performance

---

## Troubleshooting

### Build Issues

**CMake can't find compiler**
```bash
# Specify compiler explicitly
cmake -B cmake-build-debug -DCMAKE_CXX_COMPILER=g++
```

**Linker errors**
- Ensure all source files are listed in `CMakeLists.txt`
- Check C++17 support: `g++ --version` (GCC 7+ required)

### Runtime Issues

**Config file not found**
```
[config] Cannot open hyperparameters.txt
```
- Ensure `hyperparameters.txt` exists in project root OR `cmake-build-debug/..`
- Or provide explicit path: `./mycelium ../hyperparameters.txt`

**No growth occurs (nodes stuck at 1-2)**
- Increase `GROW_MULTIPLIER` (try 3.0+)
- Decrease `THRESHOLD_SPROUT` (try 0.2)
- Reduce `WALL_SAFETY_MARGIN` (try 0.01-0.02)
- Enable debug: `DEBUG_GROW = 1` to diagnose

**Nodes entrapped in walls**
- Reduce `WALL_STUCK_THRESHOLD` (try 0.3)
- Increase `WALL_UNSTUCK_FORCE` (try 1.5)
- Reduce `R_MIN` (try 0.2)

**Simulation too slow**
- Disable debug logging: `DEBUG_GROW = 0`, `DEBUG_SHIFT = 0`
- Reduce step count in `main.cpp`
- Increase raycast/check step sizes

---

## References

- Fukasawa, Y., et al. (2024). Spatial recognition in fungal mycelium networks.
- Money, N. P. (2021). Hyphal and mycelial mechanics. *Fungal Biology Reviews*.

---
