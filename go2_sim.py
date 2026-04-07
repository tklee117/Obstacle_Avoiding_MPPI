"""
go2_sim.py — MPPI high-level planner + walk-these-ways locomotion policy
running on the Unitree Go2 MuJoCo model (mujoco_menagerie).

Architecture
------------
  MPPI (10 Hz) ──► velocity commands [vx, vy, yaw_rate]
                        │
  Locomotion policy (50 Hz) ──► joint position targets
                        │
  MuJoCo physics (500 Hz) ──► simulation state

Usage
-----
  python go2_sim.py [--max-time 20] [--output-dir output] [--render]

Requirements
------------
  pip install mujoco imageio matplotlib torch numpy
"""

# Force EGL (GPU offscreen) before any mujoco / OpenGL import.
# This is required on headless servers; safe to set even with a display.
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import os
import sys
import time

import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use("Agg")   # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project imports ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mppi_hard_constraint import Obstacle, MPPISimulation
from mppi_go2 import MPPIGo2Controller
from global_planner import AStarPlanner

# ── paths ────────────────────────────────────────────────────────────────────
_HOME = os.path.expanduser("~")

MODEL_PATH   = os.path.join(_HOME, "mujoco_menagerie/unitree_go2/scene.xml")
POLICY_DIR   = os.path.join(
    _HOME,
    "walk-these-ways/runs/gait-conditioned-agility/pretrain-v0"
    "/train/025417.456545/checkpoints",
)

# ── policy & physics constants (from walk-these-ways parameters.pkl) ─────────

# Default standing joint angles in policy order:
#   FL_hip, FL_thigh, FL_calf,  FR_hip, FR_thigh, FR_calf,
#   RL_hip, RL_thigh, RL_calf,  RR_hip, RR_thigh, RR_calf
DEFAULT_DOF_POS = np.array([
     0.1,  0.8, -1.5,   # FL
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
], dtype=np.float32)

# Hip joint indices in the 12-dim action vector
HIP_INDICES = np.array([0, 3, 6, 9])

P_GAIN             = 20.0   # Nm/rad  (stiffness["joint"])
D_GAIN             =  0.5   # Nm·s/rad  (damping["joint"])
ACTION_SCALE       =  0.25
HIP_SCALE_REDUCTION = 0.5
CLIP_ACTIONS       = 10.0   # from normalization.clip_actions
CLIP_OBS           = 100.0  # from normalization.clip_observations

# Observation scales
OBS_SCALES = dict(
    lin_vel              = 2.0,
    ang_vel              = 0.25,
    dof_pos              = 1.0,
    dof_vel              = 0.05,
    body_height_cmd      = 2.0,
    footswing_height_cmd = 0.15,
    body_pitch_cmd       = 0.3,
    body_roll_cmd        = 0.3,
    stance_width_cmd     = 1.0,
    stance_length_cmd    = 1.0,
    aux_reward_cmd       = 1.0,
)

# Scale vector for the 15-command obs slice (matches lcm_agent.py commands_scale)
COMMANDS_SCALE = np.array([
    OBS_SCALES["lin_vel"],              # [0]  vx
    OBS_SCALES["lin_vel"],              # [1]  vy
    OBS_SCALES["ang_vel"],              # [2]  yaw_rate
    OBS_SCALES["body_height_cmd"],      # [3]  body_height
    1.0,                                # [4]  gait_frequency
    1.0,                                # [5]  gait_phase
    1.0,                                # [6]  gait_offset
    1.0,                                # [7]  gait_bound
    1.0,                                # [8]  gait_duration
    OBS_SCALES["footswing_height_cmd"], # [9]  footswing_height
    OBS_SCALES["body_pitch_cmd"],       # [10] body_pitch
    OBS_SCALES["body_roll_cmd"],        # [11] body_roll
    OBS_SCALES["stance_width_cmd"],     # [12] stance_width
    OBS_SCALES["stance_length_cmd"],    # [13] stance_length
    OBS_SCALES["aux_reward_cmd"],       # [14] aux_reward
], dtype=np.float32)

# Default gait configuration (trot)
DEFAULT_GAIT = np.array([
    0.0,   # [3]  body_height_cmd — neutral
    3.0,   # [4]  gait_frequency  — 3 Hz trot
    0.0,   # [5]  gait_phase      — trot front-left ref
    0.5,   # [6]  gait_offset     — trot: back 0.5 cycle behind
    0.0,   # [7]  gait_bound
    0.5,   # [8]  gait_duration
    0.08,  # [9]  footswing_height — 8 cm
    0.0,   # [10] body_pitch
    0.0,   # [11] body_roll
    0.25,  # [12] stance_width — 25 cm
    0.40,  # [13] stance_length — 40 cm
    0.0,   # [14] aux_reward_cmd
], dtype=np.float32)

# Simulation frequencies
SIM_DT        = 0.002   # s  (Go2 scene.xml default)
SIM_FREQ      = int(1.0 / SIM_DT)   # 500 Hz
LOCO_FREQ     = 50      # Hz — locomotion policy
MPPI_FREQ     = 10      # Hz — MPPI replan
STEPS_PER_LOCO = SIM_FREQ // LOCO_FREQ   # 10
STEPS_PER_MPPI = SIM_FREQ // MPPI_FREQ   # 50

# Observation history
NUM_OBS         = 70
NUM_OBS_HISTORY = 30
NUM_OBS_TOTAL   = NUM_OBS * NUM_OBS_HISTORY  # 2100


# ── helper functions ─────────────────────────────────────────────────────────

def quat_to_rot_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix (body → world)."""
    w, x, y, z = quat_wxyz.astype(float)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """Quaternion [w, x, y, z] → yaw angle (rad)."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))


def build_obs(data: mujoco.MjData,
              commands: np.ndarray,
              actions: np.ndarray,
              last_actions: np.ndarray,
              clock_inputs: np.ndarray) -> np.ndarray:
    """
    Build the 70-dimensional observation vector consumed by the policy.

    Layout (matches walk-these-ways lcm_agent.py get_obs()):
      gravity_body   (3)
      commands_scaled (15)
      dof_pos_delta  (12)
      dof_vel_scaled (12)
      actions        (12)   — current clipped action
      last_actions   (12)   — observe_two_prev_actions
      clock_inputs   (4)    — observe_clock_inputs
    """
    quat = data.qpos[3:7]          # [w, x, y, z]
    R = quat_to_rot_mat(quat)      # body → world

    # Project gravity into body frame
    gravity_body = R.T @ np.array([0.0, 0.0, -1.0])

    # Joint states (qpos[7:19] / qvel[6:18] match policy order FL/FR/RL/RR)
    joint_pos = data.qpos[7:19].astype(np.float32)
    joint_vel = data.qvel[6:18].astype(np.float32)

    obs = np.concatenate([
        gravity_body.astype(np.float32),
        (commands * COMMANDS_SCALE).astype(np.float32),
        (joint_pos - DEFAULT_DOF_POS) * OBS_SCALES["dof_pos"],
        joint_vel * OBS_SCALES["dof_vel"],
        actions,
        last_actions,
        clock_inputs,
    ])

    return np.clip(obs, -CLIP_OBS, CLIP_OBS).astype(np.float32)


# ── main simulator class ─────────────────────────────────────────────────────

def _build_scene_xml(obstacles, goal: np.ndarray) -> str:
    """
    Read the base Go2 scene XML and inject obstacle + goal geoms as static
    visual-only bodies (contype/conaffinity=0 so they don't interfere with physics).

    Each obstacle gets:
      - a solid red cylinder  (actual radius)
      - a translucent orange cylinder (inflated safety radius)

    The goal gets a flat green disc on the floor.
    """
    with open(MODEL_PATH, "r") as fh:
        xml = fh.read()

    inject = ""
    h = 0.15  # visual obstacle half-height (metres) — short enough to see the robot body above
    for i, obs in enumerate(obstacles):
        if obs.shape == "circle":
            r_phys = obs.radius
            r_safe = obs.inflated_radius
            body_geom = (
                f'<geom name="obs_{i}_body" type="cylinder" '
                f'size="{r_phys:.4f} {h:.4f}" '
                f'rgba="0.85 0.15 0.10 0.85" contype="0" conaffinity="0"/>'
            )
            safe_geom = (
                f'<geom name="obs_{i}_safe" type="cylinder" '
                f'size="{r_safe:.4f} {h:.4f}" '
                f'rgba="1.00 0.55 0.10 0.18" contype="0" conaffinity="0"/>'
            )
        else:  # rectangle → MuJoCo box geom
            hw = obs.width  / 2.0
            hh_rect = obs.height / 2.0
            sm = obs.safety_margin
            body_geom = (
                f'<geom name="obs_{i}_body" type="box" '
                f'size="{hw:.4f} {hh_rect:.4f} {h:.4f}" '
                f'rgba="0.85 0.15 0.10 0.85" contype="0" conaffinity="0"/>'
            )
            safe_geom = (
                f'<geom name="obs_{i}_safe" type="box" '
                f'size="{hw+sm:.4f} {hh_rect+sm:.4f} {h:.4f}" '
                f'rgba="1.00 0.55 0.10 0.18" contype="0" conaffinity="0"/>'
            )
        inject += f"""
    <body name="obs_{i}" pos="{obs.x} {obs.y} {h}">
      {body_geom}
      {safe_geom}
    </body>"""

    # Goal marker: flat disc on the floor
    inject += f"""
    <body name="goal_marker" pos="{goal[0]} {goal[1]} 0.02">
      <geom name="goal_disc" type="cylinder" size="0.30 0.02"
            rgba="0.10 0.85 0.15 0.80" contype="0" conaffinity="0"/>
    </body>"""

    xml = xml.replace("</worldbody>", inject + "\n  </worldbody>")
    return xml


def _load_model_with_scene(obstacles, goal: np.ndarray) -> mujoco.MjModel:
    """
    Write the modified scene XML to a temp file in the same directory as the
    original so that relative <include> paths (e.g. go2.xml) resolve correctly,
    then load and return the MjModel.
    """
    import tempfile
    xml = _build_scene_xml(obstacles, goal)
    scene_dir = os.path.dirname(MODEL_PATH)
    fd, tmp_path = tempfile.mkstemp(dir=scene_dir, suffix=".xml")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(xml)
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model


class Go2Simulator:
    """Wraps the MuJoCo Go2 model and the walk-these-ways locomotion policy."""

    def __init__(self, obstacles=None, goal: np.ndarray = None):
        # Build scene with obstacles injected (or plain scene if none given)
        if obstacles is not None and goal is not None:
            self.model = _load_model_with_scene(obstacles, goal)
        else:
            self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # Locomotion policy (two-part: adaptation module + body)
        print("Loading locomotion policy …")
        self.body_net  = torch.jit.load(
            os.path.join(POLICY_DIR, "body_latest.jit")).eval()
        self.adapt_net = torch.jit.load(
            os.path.join(POLICY_DIR, "adaptation_module_latest.jit")).eval()
        print("Policy loaded.")

        # Internal state buffers (reset by self.reset())
        self.obs_history  = np.zeros(NUM_OBS_TOTAL, dtype=np.float32)
        self.actions      = np.zeros(12, dtype=np.float32)
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.clock_inputs = np.zeros(4,  dtype=np.float32)
        self.gait_index   = 0.0

        # 15-dim command vector: first 3 set by MPPI, rest are gait defaults
        self.commands = np.zeros(15, dtype=np.float32)
        self.commands[3:] = DEFAULT_GAIT

    # ── initialisation ────────────────────────────────────────────────

    def reset(self):
        """Reset simulation to the policy's default standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.34]      # starting height
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright
        self.data.qpos[7:19] = DEFAULT_DOF_POS
        mujoco.mj_forward(self.model, self.data)

        self.obs_history[:]  = 0.0
        self.actions[:]      = 0.0
        self.last_actions[:] = 0.0
        self.clock_inputs[:] = 0.0
        self.gait_index      = 0.0
        self.commands[:3]    = 0.0
        self.commands[3:]    = DEFAULT_GAIT

    # ── per-step helpers ──────────────────────────────────────────────

    def _update_clock(self):
        """Advance gait phase and recompute clock inputs (called at LOCO_FREQ)."""
        loco_dt = 1.0 / LOCO_FREQ
        freq    = float(self.commands[4])
        phase   = float(self.commands[5])
        offset  = float(self.commands[6])
        bound   = float(self.commands[7])

        self.gait_index = (self.gait_index + loco_dt * freq) % 1.0

        # Foot phases: matches lcm_agent.py (pacing_offset=False)
        foot_phases = [
            self.gait_index + phase + offset + bound,
            self.gait_index + offset,
            self.gait_index + bound,
            self.gait_index + phase,
        ]
        for i in range(4):
            self.clock_inputs[i] = np.sin(2.0 * np.pi * foot_phases[i])

    def _run_policy(self):
        """
        One locomotion policy step:
          1. Build 70-D obs, slide history window.
          2. adaptation_module(obs_history) → latent.
          3. body(cat(obs_history, latent))  → action.
          4. Convert action → joint torques via PD control.
          5. Write torques to data.ctrl.
        """
        obs = build_obs(
            self.data, self.commands,
            self.actions, self.last_actions, self.clock_inputs,
        )

        # Slide observation history window (oldest obs dropped, new one appended)
        self.obs_history = np.roll(self.obs_history, -NUM_OBS)
        self.obs_history[-NUM_OBS:] = obs

        obs_t = torch.from_numpy(self.obs_history[np.newaxis, :])  # (1, 2100)

        with torch.no_grad():
            latent = self.adapt_net.forward(obs_t)
            action = self.body_net.forward(
                torch.cat([obs_t, latent], dim=-1))

        action_np = np.clip(action[0].numpy(), -CLIP_ACTIONS, CLIP_ACTIONS)

        # Store for next obs
        self.last_actions = self.actions.copy()
        self.actions = action_np.astype(np.float32)

        # Action → joint position target (mirrors lcm_agent.py publish_action)
        a_scaled = action_np * ACTION_SCALE
        a_scaled[HIP_INDICES] *= HIP_SCALE_REDUCTION
        joint_target = DEFAULT_DOF_POS + a_scaled

        # PD torques
        joint_pos = self.data.qpos[7:19]
        joint_vel = self.data.qvel[6:18]
        torques = P_GAIN * (joint_target - joint_pos) + D_GAIN * (0.0 - joint_vel)
        torques = np.clip(torques, -45.0, 45.0)

        self.data.ctrl[:12] = torques

    # ── main simulation loop ──────────────────────────────────────────

    def run(
        self,
        obstacles,
        goal: np.ndarray,
        mppi: MPPIGo2Controller,
        max_time: float = 20.0,
        warmup_time: float = 2.0,
        output_dir: str = "output",
        render: bool = False,
    ):
        """
        Run the full MPPI + locomotion policy loop.

        Parameters
        ----------
        obstacles   : list of Obstacle
        goal        : (2,) target xy position
        mppi        : MPPIGo2Controller instance
        max_time    : simulation wall-time limit (seconds)
        warmup_time : seconds of standing still before MPPI activates
        output_dir  : where to save plots / video
        render      : if True, attempt to collect offscreen frames for a video

        Returns
        -------
        traj_xy   : (N, 2) array of (x, y) positions at 10 Hz
        goal_reached : bool
        """
        self.reset()
        os.makedirs(output_dir, exist_ok=True)

        max_steps    = int(max_time * SIM_FREQ)
        warmup_steps = int(warmup_time * SIM_FREQ)

        # Logging
        traj_xy  = []
        cmd_log  = []
        cbf_log  = []   # per-step CBF correction: [Δcmd_vx, Δcmd_vy, Δcmd_yaw]

        # Current MPPI command (updated at MPPI_FREQ)
        cmd_vx = cmd_vy = cmd_yaw = 0.0

        # CBF correction for the current step (zero during warmup)
        cbf_delta = np.zeros(3)

        # Optional offscreen renderer (EGL backend set at module top)
        renderer = None
        cam      = None
        frames   = []
        if render:
            try:
                renderer = mujoco.Renderer(self.model, height=480, width=640)
                # Free camera: isometric perspective that tracks the robot
                cam = mujoco.MjvCamera()
                cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
                cam.distance  = 10.0   # wide enough to see the full arena
                cam.elevation = -45.0  # degrees below horizontal
                cam.azimuth   = 135.0  # north-east view
                cam.lookat[:] = [3.0, 3.0, 0.5]  # fixed on arena centre
                print("Offscreen renderer ready (EGL).")
            except Exception as e:
                print(f"Renderer unavailable: {e}")
                renderer = None

        goal_reached = False
        rollout_history = []   # list of dicts — one entry per MPPI replan
        t_start = time.time()

        # ── global A* path ────────────────────────────────────────────────────
        start_pos = np.array([self.data.qpos[0], self.data.qpos[1]])
        all_x = [start_pos[0], goal[0]] + [o.x for o in obstacles]
        all_y = [start_pos[1], goal[1]] + [o.y for o in obstacles]
        bounds = (
            min(all_x) - 1.5, max(all_x) + 1.5,
            min(all_y) - 1.5, max(all_y) + 1.5,
        )
        astar = AStarPlanner(
            obstacles, bounds,
            resolution=0.15,
            safety_margin=mppi.safety_margin,
        )
        global_path = astar.plan(start_pos, goal)
        if global_path is None:
            print("WARNING: A* found no path — falling back to direct goal tracking")
            global_path = [start_pos, goal]
        CARROT_LOOKAHEAD = 1.5   # metres ahead of robot on global path

        print(f"Starting simulation (warmup={warmup_time}s, max={max_time}s) …")
        print(f"Goal: {goal}   Obstacles: {len(obstacles)}")

        for step in range(max_steps):
            t_sim = step * SIM_DT

            # ── MPPI replan (10 Hz, after warmup) ────────────────────────────
            if step >= warmup_steps and step % STEPS_PER_MPPI == 0:
                pos_x = float(self.data.qpos[0])
                pos_y = float(self.data.qpos[1])
                yaw   = quat_to_yaw(self.data.qpos[3:7])
                vel_x = float(self.data.qvel[0])
                vel_y = float(self.data.qvel[1])

                # Pure-pursuit carrot on the global path
                robot_pos = np.array([pos_x, pos_y])
                carrot = astar.get_carrot(global_path, robot_pos, CARROT_LOOKAHEAD)

                state  = np.array([pos_x, pos_y, yaw, vel_x, vel_y])
                u_mppi, _, _ = mppi.update_control(state, carrot, obstacles)
                u_opt  = mppi.apply_cbf_filter(state, u_mppi, obstacles)
                cbf_delta = u_opt - u_mppi          # how much CBF modified the command
                cmd_vx, cmd_vy, cmd_yaw = u_opt

                # Capture rollout snapshot for the 2-D animation
                rollout_history.append({
                    "trajectories": [t.copy() for t in mppi.last_sampled_trajectories],
                    "weights":      mppi.last_weights.copy(),
                    "current_pos":  np.array([pos_x, pos_y]),
                })

                # Feed velocity commands into the 15-dim command vector
                self.commands[0] = float(np.clip(cmd_vx,  -1.5, 1.5))
                self.commands[1] = float(np.clip(cmd_vy,  -0.8, 0.8))
                self.commands[2] = float(np.clip(cmd_yaw, -1.0, 1.0))

                dist = float(np.hypot(pos_x - goal[0], pos_y - goal[1]))

                if dist < 0.3:
                    print(f"Goal reached at t={t_sim:.1f}s!  dist={dist:.3f}m")
                    goal_reached = True
                    traj_xy.append(np.array([pos_x, pos_y]))
                    cmd_log.append(np.array([cmd_vx, cmd_vy, cmd_yaw]))
                    cbf_log.append(cbf_delta.copy())
                    break

                # Periodic progress print
                if step % (STEPS_PER_MPPI * 10) == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"  t={t_sim:.1f}s  pos=({pos_x:.2f},{pos_y:.2f})"
                        f"  dist={dist:.2f}m  cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_yaw:.2f})"
                        f"  wall={elapsed:.1f}s"
                    )

            # ── locomotion policy step (50 Hz) ────────────────────────────────
            if step % STEPS_PER_LOCO == 0:
                self._update_clock()
                self._run_policy()

            # ── physics step ──────────────────────────────────────────────────
            mujoco.mj_step(self.model, self.data)

            # ── logging / rendering at 10 Hz ──────────────────────────────────
            if step % STEPS_PER_MPPI == 0:
                traj_xy.append(np.array([self.data.qpos[0], self.data.qpos[1]]))
                cmd_log.append(np.array([cmd_vx, cmd_vy, cmd_yaw]))
                cbf_log.append(cbf_delta.copy())

                if renderer is not None:
                    try:
                        renderer.update_scene(self.data, camera=cam)
                        frames.append(renderer.render().copy())
                    except Exception:
                        renderer = None  # disable on first failure

        total_wall = time.time() - t_start
        traj_xy  = np.array(traj_xy)  if traj_xy  else np.zeros((1, 2))
        cmd_log  = np.array(cmd_log)  if cmd_log  else np.zeros((1, 3))
        cbf_log  = np.array(cbf_log)  if cbf_log  else np.zeros((1, 3))

        print(f"Simulation finished in {total_wall:.1f}s real time.")
        print(f"Goal reached: {goal_reached}   Trajectory points: {len(traj_xy)}")
        cbf_active = int(np.sum(np.linalg.norm(cbf_log[:, :2], axis=1) > 1e-3))
        print(f"CBF filter active: {cbf_active}/{len(cbf_log)} steps "
              f"({100 * cbf_active / max(len(cbf_log), 1):.1f}%)")

        self._save_results(traj_xy, cmd_log, cbf_log, obstacles, goal, frames, rollout_history, output_dir, global_path)
        return traj_xy, goal_reached

    # ── output generation ─────────────────────────────────────────────

    def _save_results(self, traj_xy, cmd_log, cbf_log, obstacles, goal, frames, rollout_history, output_dir, global_path=None):
        """Save trajectory plot, rollout GIF, and (if available) MuJoCo video."""
        self._save_trajectory_plot(traj_xy, cmd_log, cbf_log, obstacles, goal, output_dir, global_path)
        if rollout_history:
            self._save_rollout_gif(traj_xy, rollout_history, obstacles, goal, output_dir, global_path=global_path)
        if frames:
            self._save_video(frames, output_dir)

    def _save_trajectory_plot(self, traj_xy, cmd_log, cbf_log, obstacles, goal, output_dir, global_path=None):
        fig, axes = plt.subplots(1, 4, figsize=(26, 6))
        ax_traj, ax_cmd, ax_dist, ax_cbf = axes

        # ── trajectory ────────────────────────────────────────────────
        ax_traj.set_title("MPPI + Go2 Navigation (MuJoCo)")
        ax_traj.set_xlabel("X (m)")
        ax_traj.set_ylabel("Y (m)")
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)

        for i, obs in enumerate(obstacles):
            MPPISimulation._draw_obstacle_patches(
                ax_traj, obs,
                show_safety=True,
                label_obs="Obstacle"    if i == 0 else "",
                label_safety="Safety zone" if i == 0 else "",
            )

        if global_path is not None:
            gp = np.array(global_path)
            ax_traj.plot(gp[:, 0], gp[:, 1], "c--", linewidth=1.5, alpha=0.7, label="A* path")
        ax_traj.plot(traj_xy[:, 0], traj_xy[:, 1], "b-", linewidth=2, label="Executed path")
        ax_traj.plot(traj_xy[0, 0],  traj_xy[0, 1],  "go", markersize=10, label="Start")
        ax_traj.plot(goal[0],         goal[1],         "r*", markersize=15, label="Goal")
        ax_traj.legend(loc="upper left")

        # Auto-scale axes with some padding
        margin = 1.0
        all_x = np.concatenate([traj_xy[:, 0], [o.x for o in obstacles], [goal[0]]])
        all_y = np.concatenate([traj_xy[:, 1], [o.y for o in obstacles], [goal[1]]])
        ax_traj.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax_traj.set_ylim(all_y.min() - margin, all_y.max() + margin)

        # ── velocity commands ─────────────────────────────────────────
        t_axis = np.arange(len(cmd_log)) * (STEPS_PER_MPPI * SIM_DT)
        ax_cmd.set_title("MPPI Velocity Commands")
        ax_cmd.set_xlabel("Time (s)")
        ax_cmd.set_ylabel("Velocity")
        ax_cmd.grid(True, alpha=0.3)
        if len(cmd_log) > 0:
            ax_cmd.plot(t_axis, cmd_log[:, 0], label="cmd_vx",  linewidth=2)
            ax_cmd.plot(t_axis, cmd_log[:, 1], label="cmd_vy",  linewidth=2)
            ax_cmd.plot(t_axis, cmd_log[:, 2], label="cmd_yaw", linewidth=2, linestyle="--")
        ax_cmd.legend()

        # ── distance to goal ──────────────────────────────────────────
        dists = np.linalg.norm(traj_xy - goal[np.newaxis, :], axis=1)
        ax_dist.set_title("Distance to Goal")
        ax_dist.set_xlabel("Time (s)")
        ax_dist.set_ylabel("Distance (m)")
        ax_dist.grid(True, alpha=0.3)
        ax_dist.plot(t_axis[:len(dists)], dists, "g-", linewidth=2)
        ax_dist.axhline(y=0.3, color="r", linestyle="--", alpha=0.7, label="Goal threshold (0.3m)")
        ax_dist.legend()

        # ── CBF filter activation ─────────────────────────────────────
        # cbf_log[:, :2] = [Δcmd_vx, Δcmd_vy] — the correction applied to the
        # translational velocity commands.  A non-zero norm means the CBF QP
        # overrode MPPI to keep the robot safe.
        t_cbf = np.arange(len(cbf_log)) * (STEPS_PER_MPPI * SIM_DT)
        cbf_norm = np.linalg.norm(cbf_log[:, :2], axis=1)   # (N,) scalar correction
        active   = cbf_norm > 1e-3                            # boolean mask
        n_active = int(np.sum(active))
        pct      = 100.0 * n_active / max(len(cbf_norm), 1)

        ax_cbf.set_title(f"CBF Filter Activation  ({n_active}/{len(cbf_norm)} steps, {pct:.1f}%)")
        ax_cbf.set_xlabel("Time (s)")
        ax_cbf.set_ylabel("Velocity correction ‖Δu‖ (m/s)")
        ax_cbf.grid(True, alpha=0.3)

        ax_cbf.plot(t_cbf, cbf_norm, color="purple", linewidth=2, label="‖Δu‖ (CBF correction)")
        ax_cbf.fill_between(t_cbf, cbf_norm, where=active,
                            color="purple", alpha=0.25, label="CBF active")

        # Overlay min SDF to any obstacle — shows *why* CBF fires
        if len(traj_xy) > 0 and len(obstacles) > 0:
            min_sdf = np.array([
                min(obs.distance_from_surface(p) for obs in obstacles)
                for p in traj_xy
            ])
            safety_margin = obstacles[0].safety_margin
            ax_sdf = ax_cbf.twinx()
            ax_sdf.plot(t_axis[:len(min_sdf)], min_sdf,
                        color="orange", linewidth=1.5, linestyle="--",
                        alpha=0.8, label="Min SDF (m)")
            ax_sdf.axhline(y=0.0, color="red", linewidth=1.0, linestyle=":",
                           alpha=0.8, label="Obstacle surface (SDF=0)")
            ax_sdf.axhline(y=safety_margin, color="orange", linewidth=1.0,
                           linestyle=":", alpha=0.8,
                           label=f"Safety zone edge (SDF={safety_margin}m)")
            ax_sdf.set_ylabel("SDF to nearest obstacle (m)", color="orange")
            ax_sdf.tick_params(axis="y", labelcolor="orange")
            # Merge legends from both axes
            h1, l1 = ax_cbf.get_legend_handles_labels()
            h2, l2 = ax_sdf.get_legend_handles_labels()
            ax_cbf.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
        else:
            ax_cbf.legend(loc="upper right")

        plt.tight_layout()
        path = os.path.join(output_dir, "go2_mppi_result.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    def _save_rollout_gif(self, traj_xy, rollout_history, obstacles, goal,
                          output_dir, max_rollouts: int = 50, global_path=None):
        """
        Create a 2-D top-down GIF showing how MPPI sampled trajectories evolve
        over time — analogous to create_rollout_animation in mppi_hard_constraint.py.

        Each frame = one MPPI replan step.
        Trajectories are coloured by weight (plasma); the executed path grows in green.
        """
        import imageio
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(8, 8))

        # Determine plot bounds from obstacles + goal + trajectory
        all_x = np.concatenate([traj_xy[:, 0], [o.x for o in obstacles], [goal[0]]])
        all_y = np.concatenate([traj_xy[:, 1], [o.y for o in obstacles], [goal[1]]])
        pad = 1.5
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Static elements: obstacles and goal
        for i, obs in enumerate(obstacles):
            MPPISimulation._draw_obstacle_patches(
                ax, obs, show_safety=True,
                label_obs="Obstacle"     if i == 0 else "",
                label_safety="Safety zone" if i == 0 else "",
            )
        ax.plot(goal[0], goal[1], "r*", markersize=14, label="Goal", zorder=6)
        if global_path is not None:
            gp = np.array(global_path)
            ax.plot(gp[:, 0], gp[:, 1], "c--", linewidth=1.5, alpha=0.7, label="A* path", zorder=4)

        # Dynamic line handles — pre-allocate rollout lines
        rollout_lines = [ax.plot([], [], alpha=0.4, linewidth=1)[0]
                         for _ in range(max_rollouts)]
        (robot,) = ax.plot([], [], "ko", markersize=10,
                           markerfacecolor="white", markeredgewidth=2.5,
                           label="Robot", zorder=7)
        (exec_path,) = ax.plot([], [], "g-", linewidth=3, alpha=0.85,
                               label="Executed path", zorder=5)

        ax.legend(loc="upper left", fontsize=8)
        title = ax.set_title("")

        # traj_xy includes warmup points; rollout_history starts after warmup.
        # Compute offset so exec_path indices align with rollout_history frames.
        warmup_offset = len(traj_xy) - len(rollout_history)

        def _animate(frame):
            data       = rollout_history[frame]
            trajs      = data["trajectories"]
            weights    = data["weights"]
            cur_pos    = data["current_pos"]

            # Pick up to max_rollouts trajectories, sorted by weight (low→high)
            idx = np.argsort(weights)
            if len(idx) > max_rollouts:
                step = len(idx) // max_rollouts
                idx = idx[::step]

            colors = plt.cm.plasma(np.linspace(0, 1, len(idx)))
            w_max  = np.max(weights) if np.max(weights) > 0 else 1.0

            for i, line in enumerate(rollout_lines):
                if i < len(idx):
                    traj = trajs[idx[i]]
                    w    = weights[idx[i]]
                    line.set_data(traj[:, 0], traj[:, 1])
                    line.set_color(colors[i])
                    line.set_alpha(0.15 + 0.65 * (w / w_max))
                else:
                    line.set_data([], [])

            robot.set_data([cur_pos[0]], [cur_pos[1]])

            # Show all warmup points plus every executed position up to and
            # including the current frame so the path stays under the robot dot.
            end = warmup_offset + frame + 1
            exec_path.set_data(traj_xy[:end, 0], traj_xy[:end, 1])

            eff = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0
            title.set_text(
                f"MPPI Rollouts — step {frame}/{len(rollout_history)-1}"
                f"   ESS={eff:.1f}/{len(weights)}"
            )
            return rollout_lines + [robot, exec_path, title]

        anim = FuncAnimation(
            fig, _animate,
            frames=len(rollout_history),
            interval=200, blit=False, repeat=True,
        )

        gif_path = os.path.join(output_dir, "go2_mppi_rollouts.gif")
        try:
            print("Saving rollout GIF …")
            # Render each frame to a numpy array and write with imageio
            frames_out = []
            for i in range(len(rollout_history)):
                _animate(i)
                fig.canvas.draw()
                w_px, h_px = fig.canvas.get_width_height()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frames_out.append(img.reshape(h_px, w_px, 3))
            imageio.mimsave(gif_path, frames_out, fps=10, loop=0)
            print(f"Saved rollout GIF: {gif_path}")
        except Exception as e:
            print(f"Rollout GIF save failed: {e}")
        finally:
            plt.close(fig)

    def _save_video(self, frames, output_dir):
        import imageio

        # GIF — always works with imageio + pillow (no ffmpeg needed)
        gif_path = os.path.join(output_dir, "go2_mppi_sim.gif")
        try:
            # Frames are captured at 10 Hz; play at 10 fps → real-time speed,
            # matching the rollout GIF frame rate exactly.
            imageio.mimsave(gif_path, frames, fps=10, loop=0)
            print(f"Saved GIF: {gif_path}")
        except Exception as e:
            print(f"GIF save failed: {e}")

        # MP4 — requires imageio[ffmpeg]
        mp4_path = os.path.join(output_dir, "go2_mppi_sim.mp4")
        try:
            imageio.mimsave(mp4_path, frames, fps=10)
            print(f"Saved MP4: {mp4_path}")
        except Exception as e:
            print(f"MP4 save failed (install imageio[ffmpeg]): {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MPPI + Go2 MuJoCo simulation")
    p.add_argument("--max-time",    type=float, default=20.0,
                   help="Maximum simulation time in seconds (default: 20)")
    p.add_argument("--warmup-time", type=float, default=2.0,
                   help="Standing warmup before MPPI activates (default: 2)")
    p.add_argument("--output-dir",  default=os.path.join(_HERE, "output"),
                   help="Directory for saved plots/video (default: output/)")
    p.add_argument("--render",      action="store_true",
                   help="Capture offscreen frames and save as gif/mp4")
    p.add_argument("--samples",     type=int,   default=500,
                   help="MPPI sample count (default: 500)")
    p.add_argument("--horizon",     type=int,   default=30,
                   help="MPPI horizon (default: 30)")
    p.add_argument("--sigma",       type=float, default=0.9,
                   help="MPPI noise sigma (default: 0.9)")
    # ── obstacle configuration ────────────────────────────────────────────────
    p.add_argument("-n", "--num-obstacles", type=int, default=None, metavar="N",
                   help="Number of obstacles (default: random 3-8 each run)")
    p.add_argument("-s", "--shape",
                   choices=["circle", "rectangle", "mixed"], default="circle",
                   help="Obstacle shape: circle, rectangle, or mixed (default: circle)")
    p.add_argument("--safety-margin", type=float, default=0.5, metavar="M",
                   help="Safety margin in metres added to every obstacle (default: 0.5)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible obstacle layouts")
    p.add_argument("--cbf-gamma", type=float, default=1.0, metavar="G",
                   help="CBF decay-rate γ: larger = more conservative filter (default: 1.0)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # ── scene definition ──────────────────────────────────────────────────────
    goal = np.array([6.0, 6.0])

    obstacles = MPPISimulation.generate_random_obstacles(
        num_obstacles  = args.num_obstacles,
        shape          = args.shape,
        safety_margin  = args.safety_margin,
        start          = np.array([0.0, 0.0]),
        goal           = goal,
        x_range        = (0.5, 5.5),
        y_range        = (0.5, 5.5),
        radius_range   = (0.3, 0.7),
        min_clearance  = 1.0,
    )

    print(f"Obstacles ({len(obstacles)}):")
    for i, obs in enumerate(obstacles):
        if obs.shape == "circle":
            desc = f"radius={obs.radius:.2f}"
        else:
            desc = f"width={obs.width:.2f}, height={obs.height:.2f}"
        print(f"  {i+1}. [{obs.shape}] pos=({obs.x:.2f},{obs.y:.2f})  {desc}  margin={obs.safety_margin}")

    # ── MPPI controller ───────────────────────────────────────────────────────
    mppi = MPPIGo2Controller(
        horizon        = args.horizon,
        num_samples    = args.samples,
        dt             = STEPS_PER_MPPI * SIM_DT,  # 0.1 s
        lambda_        = 7.0,
        sigma          = args.sigma,
        control_bounds = (-1.5, 1.5),
        safety_margin  = args.safety_margin,
        cbf_gamma      = args.cbf_gamma,
    )

    # ── run ───────────────────────────────────────────────────────────────────
    sim = Go2Simulator(obstacles=obstacles, goal=goal)
    traj, goal_reached = sim.run(
        obstacles   = obstacles,
        goal        = goal,
        mppi        = mppi,
        max_time    = args.max_time,
        warmup_time = args.warmup_time,
        output_dir  = args.output_dir,
        render      = args.render,
    )

    print()
    print("=" * 50)
    print(f"Goal reached      : {goal_reached}")
    final_dist = float(np.linalg.norm(traj[-1] - goal))
    print(f"Final distance    : {final_dist:.3f} m")
    print(f"Trajectory points : {len(traj)}")
    print(f"Output saved to   : {os.path.abspath(args.output_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
