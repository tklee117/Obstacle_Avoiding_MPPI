# MPPI Obstacle Avoidance with Safety Margins

**Model Predictive Path Integral (MPPI) control for autonomous robot navigation** — including a high-level planner for the Unitree Go2 quadruped running in MuJoCo.

---

## Overview

This project implements and evaluates MPPI control for mobile robot navigation in environments with static obstacles. The core idea: instead of solving a constrained optimization problem, MPPI samples thousands of candidate trajectories in parallel, weights them by cost (via importance sampling), and blends them into an optimal control update — all without needing gradients.

Safety margins are implemented through **obstacle inflation**: every obstacle is expanded by a configurable buffer distance, so the planner naturally steers clear of boundaries rather than just avoiding hard collisions.

The project includes two main applications:

1. **2D point-mass simulation** — fast, fully self-contained, used for benchmarking and parameter sweeps
2. **Unitree Go2 MuJoCo simulation** — MPPI high-level planner feeding velocity commands to a learned locomotion policy (walk-these-ways)

---

## Repository Structure

```
.
├── mppi_hard_constraint.py   # Core MPPI controller + 2D simulation environment
├── mppi_go2.py               # Go2-specific MPPI (unicycle dynamics, velocity commands)
├── go2_sim.py                # MuJoCo simulation loop + locomotion policy integration
├── global_planner.py         # A* global planner for collision-free waypoint paths
├── mppi_performance.py       # Benchmark suite: 20-scenario statistical testing
├── mppi_test.py              # Extensive parameter sensitivity testing framework
└── output/                   # Saved plots, GIFs, and videos
```

---

## Architecture

```
A* Global Planner (once at startup)
    │  builds 2-D occupancy grid from obstacle SDFs
    │  finds collision-free waypoint path: start → goal
    │  provides pure-pursuit carrot point at runtime
    ▼
MPPI Local Planner (10 Hz)
    │  receives carrot point (not raw goal) from A*
    │  samples N control sequences
    │  rolls out trajectories via dynamics model
    │  evaluates cost (goal + obstacles + velocity + control effort)
    │  importance-weights and blends → optimal velocity command
    ▼
Locomotion Policy (50 Hz)          [Go2 only]
    │  adaptation module + body network (walk-these-ways)
    │  maps velocity commands → joint torque targets
    ▼
MuJoCo Physics (500 Hz)           [Go2 only]
    │  steps rigid-body simulation
    ▼
Robot State → back to MPPI
```

### Why A* + MPPI?

MPPI is a local optimizer — it samples perturbations around the current nominal trajectory and cannot reliably escape local minima. In front of a wall, all forward trajectories incur high obstacle cost while all sideways trajectories incur high terminal goal cost. The weighted average collapses to near-zero velocity: the robot gets stuck.

A* solves the **topological** problem (which side of the wall to go around) once, globally. MPPI solves the **local execution** problem (smooth, dynamically-feasible tracking) at 10 Hz. The carrot point from A* moves along the precomputed path as the robot progresses, so MPPI only ever sees a short-range, unobstructed subgoal.

For the **2D point-mass** case, the dynamics are a double integrator:

```
ẋ = [vx, vy, ax, ay]    control = [ax, ay]
```

For the **Go2**, the dynamics are a unicycle model:

```
state  = [x, y, yaw, vx_world, vy_world]
control = [cmd_vx, cmd_vy, cmd_yaw_rate]   (body frame)
```

---

## Cost Function

At each planning step, every sampled trajectory is scored by:

| Term | Formula | Purpose |
|---|---|---|
| Terminal goal cost | `Q_goal · ‖p_H − p_goal‖²` | Drive toward goal at horizon end |
| Running goal cost | `Q_goal_running · ‖p_t − p_goal‖²` per step | Break local minima; reward progress at every step |
| Obstacle penalty | `Q_obs · max(0, r_i + d_safety − ‖p − c_i‖)²` | Repel from inflated obstacles |
| Collision penalty | `10000 + 2000 · exp(…)` if `dist < 0` | Hard collision deterrent |
| Velocity cost | `Q_vel · ‖v‖²` | Regulate speed |
| Control effort | `R · ‖u‖²` | Smooth inputs |

The **running goal cost** was added to supplement the terminal-only goal cost. Without it, a robot stuck in front of a wall sees symmetric cost in all directions (forward = obstacle penalty, sideways = high terminal cost) and the weighted average of samples collapses to zero velocity.

Importance weights are computed via softmax with temperature λ:

```
w_i = exp(−S_i / λ) / Σ exp(−S_j / λ)
```

The control sequence update is the weighted average of all sampled sequences.

---

## Key Parameters

### MPPI

| Parameter | Symbol | Default (Go2) | Effect |
|---|---|---|---|
| Horizon | H | 30 steps (3 s) | Lookahead distance |
| Samples | N | 500 | Solution quality vs. compute |
| Noise std dev | σ | 0.9 | Exploration breadth |
| Temperature | λ | 7.0 | Weight diversity; higher = keeps more trajectories alive |
| Safety margin | d_safety | 0.5 m | Buffer around obstacles |
| Running goal weight | Q_goal_running | 1.0 | Per-step progress toward goal |
| Velocity weight | Q_vel | 0.05 | Resistance to lateral velocity bursts |

**Key findings from parameter sweeps:**
- Higher σ (more noise) *improves* success rate in cluttered environments — aggressive exploration helps escape local minima
- Lower λ (sharper exploitation) consistently outperforms higher values in open environments; higher λ is needed to keep sideways detour trajectories alive near walls
- Moderate sample counts (100–500) match the performance of 1000+ samples at a fraction of the compute cost

### A* Global Planner

| Parameter | Default | Effect |
|---|---|---|
| Grid resolution | 0.15 m | Finer = more accurate path, slower build |
| Safety margin | matches MPPI | Inflates obstacles consistently with the local planner |
| Carrot lookahead | 1.5 m | How far ahead of the robot the MPPI subgoal is placed |

---

## Installation

```bash
# Core dependencies
pip install numpy matplotlib

# For MuJoCo / Go2 simulation
pip install mujoco torch imageio

# For benchmarking / parameter testing
pip install pandas seaborn
```

**Go2 simulation additionally requires:**
- [`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie) — place at `~/mujoco_menagerie/`
- [`walk-these-ways`](https://github.com/Improbable-AI/walk-these-ways) pretrained checkpoints — place at `~/walk-these-ways/runs/`

---

## Usage

### 2D Point-Mass Simulation

```bash
# Run with random obstacles (mixed shapes, default safety margin)
python mppi_hard_constraint.py

# Specify number and shape of obstacles
python mppi_hard_constraint.py -n 5 -s circle --safety-margin 0.3

# Reproducible run with fixed seed
python mppi_hard_constraint.py --seed 42
```

### Go2 MuJoCo Simulation

```bash
# Basic run
python go2_sim.py

# With rendering and custom parameters
python go2_sim.py --render --samples 500 --sigma 0.5 --max-time 30

# Circle obstacles only, fixed seed
python go2_sim.py -n 4 -s circle --seed 7
```

### Benchmarking

```bash
# Run 20-scenario benchmark (produces statistical analysis plots)
python mppi_performance.py

# Parameter sensitivity sweep
python mppi_test.py
```

---

## Results Summary

Evaluated across **20 randomized scenarios** (3–8 obstacles, varying layouts):

| Metric | Value |
|---|---|
| Success rate | 100% |
| Safety violation rate | 0.8% |
| Average computation time | 1.54 s/scenario |
| Path efficiency (actual / straight-line) | 1.0–1.2× |
| Effective samples (ESS) | ~4–5% of N |

The controller navigates complex environments with zero collisions and near-optimal path lengths, while the low ESS (~4–5%) is characteristic of importance sampling in high-cost-contrast settings — the algorithm concentrates weight on the genuinely good trajectories.

---

## Output Files

After each run, results are saved to `output/`:

| File | Description |
|---|---|
| `results.png` | Trajectory, velocity, control inputs, cost evolution |
| `safety_analysis.png` | Safety margin violations, clearance distances |
| `go2_mppi_result.png` | Go2 navigation trajectory (with A* path overlay) + command log |
| `go2_mppi_rollouts.gif` | Animated MPPI rollouts with A* path overlay (top-down view) |
| `go2_mppi_sim.gif` | MuJoCo offscreen render (if `--render` used) |

---

## Limitations & Future Work

- **Static obstacles only** — A* and MPPI both assume a static map; dynamic environments require replanning or extensions to the cost function
- **A* replanning** — the global path is computed once at startup; if the robot is significantly perturbed or a new obstacle appears, A* would need to rerun
- **Parameter sensitivity** — MPPI performance varies with σ and λ; auto-tuning based on environment density would improve robustness
- **Computational scaling** — 1000-sample configurations can exceed real-time budgets on CPU; GPU vectorization would help
- **Hardware validation** — real Go2 deployment would reveal additional practical constraints (latency, state estimation noise)

---

## References

- Williams, G. et al. "Model Predictive Path Integral Control using Covariance Variable Importance Sampling." *ICRA 2017.*
- Kumar, A. et al. "Walk These Ways: Tuning Robot Walking to Generalize Across Terrains." *CoRL 2022.*
- MuJoCo Menagerie — Unitree Go2 model.
