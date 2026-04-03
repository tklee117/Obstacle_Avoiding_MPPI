import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class Obstacle:
    """Represents an obstacle (circle or rectangle) in the environment."""

    x: float
    y: float
    radius: float                       # circle radius; ignored for rectangles
    safety_margin: float = 0.2
    shape: str = "circle"               # "circle" or "rectangle"
    width: Optional[float] = None       # rectangle full width  (shape="rectangle")
    height: Optional[float] = None      # rectangle full height (shape="rectangle")

    @property
    def inflated_radius(self) -> float:
        """Circle only: radius + safety margin."""
        return self.radius + self.safety_margin

    def distance_from_surface(self, pos: np.ndarray) -> float:
        """
        Signed distance from *pos* to the obstacle surface.
        Positive  → outside   |  Negative → inside (collision)
        """
        if self.shape == "circle":
            return float(np.linalg.norm(pos - np.array([self.x, self.y])) - self.radius)
        # Rectangle SDF (axis-aligned)
        hw = (self.width  or self.radius * 2) / 2.0
        hh = (self.height or self.radius * 2) / 2.0
        qx = abs(pos[0] - self.x) - hw
        qy = abs(pos[1] - self.y) - hh
        outside = np.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2)
        inside  = min(max(qx, qy), 0.0)
        return float(outside + inside)

    def is_collision(self, pos: np.ndarray) -> bool:
        return self.distance_from_surface(pos) < 0.0

    def is_in_safety_zone(self, pos: np.ndarray) -> bool:
        return self.distance_from_surface(pos) < self.safety_margin


class MPPIController:
    """Model Predictive Path Integral Controller with Inflated Obstacle Safety"""

    def __init__(
        self,
        horizon: int = 20,
        num_samples: int = 100,
        control_dim: int = 2,
        state_dim: int = 4,
        dt: float = 0.1,
        lambda_: float = 1.0,
        sigma: float = 1.0,
        control_bounds: Tuple[float, float] = (-2.0, 2.0),
        safety_margin: float = 0.2,
    ):
        """
        Initialize MPPI controller

        Args:
            horizon: Planning horizon (number of time steps)
            num_samples: Number of trajectory samples
            control_dim: Dimension of control input (2 for [vx, vy])
            state_dim: Dimension of state (4 for [x, y, vx, vy])
            dt: Time step
            lambda_: Temperature parameter for MPPI
            sigma: Control noise standard deviation
            control_bounds: Min/max control values
            safety_margin: Safety margin to add to all obstacles
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.dt = dt
        self.lambda_ = lambda_
        self.sigma = sigma
        self.control_bounds = control_bounds
        self.safety_margin = safety_margin

        self.U = np.zeros((self.horizon, self.control_dim))

        self.Q_goal = 50.0 
        self.Q_obstacle = (
            500.0  # Obstacle avoidance weight 
        )
        self.R = 0.1  
        self.Q_velocity = 0.5  

        self.last_sampled_trajectories = []
        self.last_weights = []
        self.last_costs = []

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simple 2D point mass dynamics: double integrator
        State: [x, y, vx, vy]
        Control: [ax, ay] (acceleration)
        """
        x, y, vx, vy = state
        ax, ay = control

        ax = np.clip(ax, self.control_bounds[0], self.control_bounds[1])
        ay = np.clip(ay, self.control_bounds[0], self.control_bounds[1])

        next_state = np.array(
            [x + vx * self.dt, y + vy * self.dt, vx + ax * self.dt, vy + ay * self.dt]
        )

        return next_state

    def rollout(
        self, initial_state: np.ndarray, control_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Rollout a trajectory given initial state and control sequence

        Returns:
            trajectory: Array of shape (horizon+1, state_dim)
        """
        trajectory = np.zeros((self.horizon + 1, self.state_dim))
        trajectory[0] = initial_state

        for t in range(self.horizon):
            trajectory[t + 1] = self.dynamics(trajectory[t], control_sequence[t])

        return trajectory

    def compute_cost(
        self,
        trajectory: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        control_sequence: np.ndarray,
    ) -> float:
        """
        Compute the cost of a trajectory using inflated obstacles

        Args:
            trajectory: State trajectory of shape (horizon+1, state_dim)
            goal: Goal position [x, y]
            obstacles: List of obstacles
            control_sequence: Control sequence of shape (horizon, control_dim)

        Returns:
            Total cost
        """
        cost = 0.0
        collision_penalty = 0.0

        # Terminal cost
        final_pos = trajectory[-1, :2]
        goal_cost = self.Q_goal * np.linalg.norm(final_pos - goal) ** 2
        cost += goal_cost

        # Running costs
        for t in range(self.horizon):
            state = trajectory[t]
            control = control_sequence[t]

            # Control effort cost
            control_cost = self.R * np.linalg.norm(control) ** 2
            cost += control_cost

            # Velocity regulation cost
            velocity = state[2:4]
            velocity_cost = self.Q_velocity * np.linalg.norm(velocity) ** 2
            cost += velocity_cost

            # Obstacle avoidance cost using signed distance field
            pos = state[:2]
            for obstacle in obstacles:
                dist = obstacle.distance_from_surface(pos)

                if dist < obstacle.safety_margin:  # inside safety zone
                    penetration = obstacle.safety_margin - dist
                    obstacle_cost = self.Q_obstacle * (penetration ** 2)
                    cost += obstacle_cost

                    # Extra exponential penalty very close to the surface
                    if dist < 0.1:
                        collision_penalty += 2000.0 * np.exp(-10 * max(dist, 0.0))

                # Massive penalty for actual collision
                if dist < 0.0:
                    collision_penalty += 10000.0

        # total collision penalty
        cost += collision_penalty

        return cost

    def check_trajectory_collision(self, trajectory: np.ndarray, obstacles: List[Obstacle]) -> bool:
        """
        Check if trajectory collides with any inflated obstacle

        Args:
            trajectory: State trajectory of shape (horizon+1, state_dim)
            obstacles: List of obstacles

        Returns:
            True if collision detected, False otherwise
        """
        for t in range(len(trajectory)):
            pos = trajectory[t, :2]
            for obstacle in obstacles:
                if obstacle.is_in_safety_zone(pos):
                    return True
        return False

    def sample_controls(self) -> np.ndarray:
        """
        Sample control sequences around the current control sequence

        Returns:
            sampled_controls: Array of shape (num_samples, horizon, control_dim)
        """
        # Generate random noise
        noise = np.random.normal(
            0, self.sigma, (self.num_samples, self.horizon, self.control_dim)
        )

        # Add noise to current control sequence
        sampled_controls = self.U[None, :, :] + noise

        # Clip controls to bounds
        sampled_controls = np.clip(
            sampled_controls, self.control_bounds[0], self.control_bounds[1]
        )

        return sampled_controls

    def update_control(self, state: np.ndarray, goal: np.ndarray, obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update the control sequence using MPPI with inflated obstacles

        Args:
            state: Current state [x, y, vx, vy]
            goal: Goal position [x, y]
            obstacles: List of obstacles

        Returns:
            Tuple of (best_control, best_trajectory, costs)
        """
        sampled_controls = self.sample_controls()

        costs = np.zeros(self.num_samples)
        trajectories = []

        for i in range(self.num_samples):
            trajectory = self.rollout(state, sampled_controls[i])
            cost = self.compute_cost(trajectory, goal, obstacles, sampled_controls[i])
            costs[i] = cost
            trajectories.append(trajectory)

        self.last_sampled_trajectories = trajectories.copy()
        self.last_costs = costs.copy()

        # Numerical stability: subtract minimum cost before exponential
        min_cost = np.min(costs)
        costs_normalized = costs - min_cost

        # Compute weights using softmax with numerical stability
        weights = np.exp(-costs_normalized / self.lambda_)
        weights_sum = np.sum(weights)

        # When all weights are zero
        if weights_sum == 0 or not np.isfinite(weights_sum):
            print("Warning: All weights are zero, using uniform weights")
            weights = np.ones(self.num_samples) / self.num_samples
        else:
            weights /= weights_sum

        # Check for NaN in weights
        if np.any(np.isnan(weights)):
            print("Warning: NaN in weights, using uniform weights")
            weights = np.ones(self.num_samples) / self.num_samples

        self.last_weights = weights.copy()

        # Weighted average of control sequences
        self.U = np.sum(weights[:, None, None] * sampled_controls, axis=0)

        # Check for NaN in control update
        if np.any(np.isnan(self.U)):
            print("Warning: NaN in control update, keeping previous control")
            self.U += np.random.normal(0, 0.1, self.U.shape)

        # Shift control sequence for next iteration (warm start)
        self.U[:-1] = self.U[1:].copy()
        self.U[-1] = np.zeros(self.control_dim)

        best_idx = np.argmin(costs)
        best_trajectory = trajectories[best_idx]

        return self.U[0], best_trajectory, costs


class MPPISimulation:
    """Simulation environment for MPPI with inflated obstacle safety margins"""

    @staticmethod
    def generate_random_obstacles(
        num_obstacles: Optional[int] = None,
        shape: str = "mixed",
        x_range: Tuple[float, float] = (0.5, 7.5),
        y_range: Tuple[float, float] = (0.5, 7.5),
        radius_range: Tuple[float, float] = (0.3, 1.0),
        safety_margin: float = 0.2,
        start: np.ndarray = None,
        goal: np.ndarray = None,
        min_clearance: float = 1.2,
    ) -> List[Obstacle]:
        """
        Generate a random set of obstacles for each simulation run.

        Args:
            num_obstacles: Number of obstacles (random 3-8 if None)
            shape: "circle", "rectangle", or "mixed" (random per obstacle)
            x_range: (min, max) x placement range
            y_range: (min, max) y placement range
            radius_range: (min, max) size — radius for circles, half-extent for rectangles
            safety_margin: Safety margin added to each obstacle
            start: Start position [x, y] — obstacles stay clear of this
            goal: Goal position [x, y] — obstacles stay clear of this
            min_clearance: Minimum gap between obstacle edge and start/goal

        Returns:
            List of Obstacle objects
        """
        if start is None:
            start = np.array([0.0, 0.0])
        if goal is None:
            goal = np.array([8.0, 8.0])
        if num_obstacles is None:
            num_obstacles = np.random.randint(3, 9)  # 3 to 8 obstacles

        valid_shapes = ("circle", "rectangle", "mixed")
        if shape not in valid_shapes:
            raise ValueError(f"shape must be one of {valid_shapes}, got '{shape}'")

        obstacles = []
        max_attempts = 300

        for _ in range(num_obstacles):
            # Pick shape for this obstacle
            obs_shape = shape if shape != "mixed" else np.random.choice(["circle", "rectangle"])

            for _ in range(max_attempts):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                r = np.random.uniform(*radius_range)  # half-extent

                # Build a candidate obstacle to use its SDF for clearance checks
                if obs_shape == "circle":
                    candidate = Obstacle(x, y, r, safety_margin, shape="circle")
                else:
                    w = r * 2 * np.random.uniform(0.8, 2.0)  # random aspect ratio
                    h = r * 2 * np.random.uniform(0.8, 2.0)
                    candidate = Obstacle(x, y, r, safety_margin, shape="rectangle",
                                         width=w, height=h)

                # Keep clear of start and goal
                if candidate.distance_from_surface(start) < min_clearance:
                    continue
                if candidate.distance_from_surface(goal) < min_clearance:
                    continue

                # Keep clear of existing obstacles (0.3 m gap between surfaces)
                overlap = any(
                    candidate.distance_from_surface(np.array([obs.x, obs.y])) < 0.3
                    for obs in obstacles
                )
                if overlap:
                    continue

                obstacles.append(candidate)
                break  # placed successfully

        return obstacles

    def __init__(
        self,
        safety_margin: float = 0.2,
        output_dir: str = "output",
        num_obstacles: Optional[int] = None,
        obstacle_shape: str = "mixed",
    ):
        """
        Args:
            safety_margin: Safety margin added to all obstacles.
            output_dir: Directory for saved plots/animations.
            num_obstacles: Number of obstacles to place (random 3-8 if None).
            obstacle_shape: "circle", "rectangle", or "mixed".
        """
        self.safety_margin = safety_margin
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.controller = MPPIController(
            horizon=20,
            num_samples=100,
            dt=0.1,
            lambda_=1.0,
            sigma=1.0,
            safety_margin=safety_margin,
        )

        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # Start at origin
        self.goal = np.array([8.0, 8.0])  # Goal position

        self.obstacles = MPPISimulation.generate_random_obstacles(
            num_obstacles=num_obstacles,
            shape=obstacle_shape,
            safety_margin=safety_margin,
            start=self.state[:2],
            goal=self.goal,
        )

        # Simulation history
        self.history = []
        self.control_history = []
        self.cost_history = []
        self.rollout_history = []
        self.safety_violations = []  # Track when robot enters safety margins

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_obstacle_patches(
        ax,
        obstacle: "Obstacle",
        show_safety: bool = False,
        label_obs: str = "",
        label_safety: str = "",
    ):
        """Add obstacle patch (and optional safety-zone patch) to *ax*."""
        if obstacle.shape == "circle":
            if show_safety:
                ax.add_patch(patches.Circle(
                    (obstacle.x, obstacle.y), obstacle.inflated_radius,
                    color="pink", alpha=0.3, label=label_safety,
                ))
            ax.add_patch(patches.Circle(
                (obstacle.x, obstacle.y), obstacle.radius,
                color="red", alpha=0.7, label=label_obs,
            ))
        else:  # rectangle
            hw = obstacle.width  / 2.0
            hh = obstacle.height / 2.0
            sm = obstacle.safety_margin
            if show_safety:
                ax.add_patch(patches.Rectangle(
                    (obstacle.x - hw - sm, obstacle.y - hh - sm),
                    obstacle.width + 2 * sm, obstacle.height + 2 * sm,
                    color="pink", alpha=0.3, label=label_safety,
                ))
            ax.add_patch(patches.Rectangle(
                (obstacle.x - hw, obstacle.y - hh),
                obstacle.width, obstacle.height,
                color="red", alpha=0.7, label=label_obs,
            ))

    def _draw_all_obstacles(self, ax, show_safety: bool = False):
        """Draw every obstacle onto *ax*, labelling only the first."""
        for i, obs in enumerate(self.obstacles):
            label_obs    = "Actual Obstacle" if i == 0 else ""
            label_safety = "Safety Margin"   if i == 0 else ""
            self._draw_obstacle_patches(ax, obs,
                                        show_safety=show_safety,
                                        label_obs=label_obs,
                                        label_safety=label_safety)

    # ------------------------------------------------------------------

    def run_simulation(self, max_steps: int = 200) -> bool:
        """
        Run the simulation with inflated obstacle safety checking

        Returns:
            True if goal reached, False if max steps exceeded
        """
        print("Starting MPPI simulation with inflated obstacle safety margins...")
        print(f"Safety margin: {self.safety_margin}")
        print(f"Initial state: {self.state}")
        print(f"Goal: {self.goal}")
        print(f"Number of obstacles: {len(self.obstacles)}")

        for i, obs in enumerate(self.obstacles):
            if obs.shape == "circle":
                desc = f"radius={obs.radius:.2f}, inflated_radius={obs.inflated_radius:.2f}"
            else:
                desc = f"width={obs.width:.2f}, height={obs.height:.2f}"
            print(f"Obstacle {i + 1} [{obs.shape}]: center=({obs.x:.1f}, {obs.y:.1f}), {desc}")

        for step in range(max_steps):
            control, best_trajectory, costs = self.controller.update_control(
                self.state, self.goal, self.obstacles
            )

            self.rollout_history.append(
                {
                    "trajectories": [
                        traj.copy()
                        for traj in self.controller.last_sampled_trajectories
                    ],
                    "weights": self.controller.last_weights.copy(),
                    "costs": self.controller.last_costs.copy(),
                    "current_state": self.state.copy(),
                }
            )

            self.state = self.controller.dynamics(self.state, control)

            # Check safety violations (robot in safety margin but not colliding)
            safety_violation = False
            actual_collision = False

            for obstacle in self.obstacles:
                dist = obstacle.distance_from_surface(self.state[:2])
                if dist < 0.0:
                    actual_collision = True
                    break
                elif dist < obstacle.safety_margin:
                    safety_violation = True

            self.safety_violations.append(safety_violation)

            self.history.append(self.state.copy())
            self.control_history.append(control.copy())
            self.cost_history.append(np.min(costs))

            distance_to_goal = np.linalg.norm(self.state[:2] - self.goal)
            if distance_to_goal < 0.3:
                print(f"Goal reached in {step + 1} steps!")
                print(
                    f"Safety violations during trajectory: {sum(self.safety_violations)} steps"
                )
                return True

            if actual_collision:
                print(f"Actual collision detected at step {step + 1}!")
                return False

            if step % 20 == 0:
                violations_so_far = sum(self.safety_violations)
                print(
                    f"Step {step}: State=[{self.state[0]:.2f}, {self.state[1]:.2f}], "
                    f"Distance to goal={distance_to_goal:.2f}, Safety violations: {violations_so_far}"
                )

        print(f"Max steps ({max_steps}) reached without reaching goal")
        print(f"Total safety violations: {sum(self.safety_violations)} steps")
        return False

    def plot_rollouts_at_timestep(self, timestep: int, max_rollouts: int = 100):
        """
        Plot all sampled rollouts at a specific timestep, showing both original and inflated obstacles
        """
        if timestep >= len(self.rollout_history):
            print(
                f"Timestep {timestep} not available. Max timestep: {len(self.rollout_history) - 1}"
            )
            return

        rollout_data = self.rollout_history[timestep]
        trajectories = rollout_data["trajectories"]
        weights = rollout_data["weights"]
        current_state = rollout_data["current_state"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        ax1.set_title(
            f"MPPI Rollouts with Safety Margins (Timestep {timestep})\n"
            f"Red=Original Obstacles, Pink=Safety Zones"
        )
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax1, show_safety=True)

        ax1.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")

        sorted_indices = np.argsort(weights)

        if len(sorted_indices) > max_rollouts:
            step = len(sorted_indices) // max_rollouts
            sorted_indices = sorted_indices[::step]

        colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_indices)))

        # Plot sampled trajectories
        collision_count = 0
        safety_violation_count = 0

        for i, idx in enumerate(sorted_indices):
            traj = trajectories[idx]
            weight = weights[idx]

            # Check if trajectory violates safety or collides
            has_collision = False
            has_safety_violation = False

            for t in range(len(traj)):
                pos = traj[t, :2]
                for obstacle in self.obstacles:
                    dist = obstacle.distance_from_surface(pos)
                    if dist < 0.0:
                        has_collision = True
                    elif dist < obstacle.safety_margin:
                        has_safety_violation = True

            if has_collision:
                collision_count += 1
                line_style = "--"
                alpha = 0.2
            elif has_safety_violation:
                safety_violation_count += 1
                line_style = "-"
                alpha = 0.3 + 0.4 * (weight / np.max(weights))
            else:
                line_style = "-"
                alpha = 0.3 + 0.7 * (weight / np.max(weights))

            ax1.plot(
                traj[:, 0],
                traj[:, 1],
                color=colors[i],
                alpha=alpha,
                linewidth=1.5,
                linestyle=line_style,
                label="Sampled Trajectories" if i == 0 else "",
            )

        # Plot current robot position
        ax1.plot(
            current_state[0],
            current_state[1],
            "ko",
            markersize=12,
            markerfacecolor="white",
            markeredgewidth=3,
            label="Current Position",
        )

        # Plot executed trajectory so far
        if self.history and timestep > 0:
            executed_traj = np.array(self.history[: timestep + 1])
            ax1.plot(
                executed_traj[:, 0],
                executed_traj[:, 1],
                "g-",
                linewidth=4,
                alpha=0.8,
                label="Executed Path",
            )

        ax1.legend()
        ax1.set_aspect("equal")

        total_trajectories = len(sorted_indices)
        ax1.text(
            0.02,
            0.98,
            f"Trajectories: {total_trajectories}\n"
            f"Collisions: {collision_count}\n"
            f"Safety Violations: {safety_violation_count}\n"
            f"Safe: {total_trajectories - collision_count - safety_violation_count}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Weight distribution analysis
        ax2.set_title(f"Weight Distribution Analysis (Timestep {timestep})")
        ax2.set_xlabel("Trajectory Index (sorted by weight)")
        ax2.set_ylabel("Weight")
        ax2.grid(True, alpha=0.3)

        sorted_weights = np.sort(weights)
        ax2.semilogy(sorted_weights, "b-", linewidth=2, label="Sorted Weights")
        ax2.fill_between(range(len(sorted_weights)), sorted_weights, alpha=0.3)

        max_weight = np.max(weights)
        mean_weight = np.mean(weights)
        effective_samples = 1.0 / np.sum(weights**2)

        ax2.axhline(
            y=max_weight,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Max Weight: {max_weight:.6f}",
        )
        ax2.axhline(
            y=mean_weight,
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Mean Weight: {mean_weight:.6f}",
        )

        ax2.text(
            0.02,
            0.98,
            f"Effective Samples: {effective_samples:.1f}/{len(weights)}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"rollouts_timestep_{timestep}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()

    def plot_results(self):
        """Plot the original simulation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Trajectory and environment
        ax1.set_title("MPPI Trajectory with Obstacle Avoidance")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax1, show_safety=False)

        # Plot trajectory
        if self.history:
            trajectory = np.array(self.history)
            ax1.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "b-",
                linewidth=2,
                label="Robot Trajectory",
            )
            ax1.plot(
                trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start"
            )

        # Plot goal
        ax1.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")
        ax1.legend()
        ax1.set_aspect("equal")

        # Plot 2: Velocity profile 
        if self.history:
            ax2.set_title("Velocity Profile")
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Velocity")
            ax2.grid(True, alpha=0.3)

            trajectory = np.array(self.history)
            ax2.plot(trajectory[:, 2], label="Vx", linewidth=2)
            ax2.plot(trajectory[:, 3], label="Vy", linewidth=2)
            ax2.plot(
                np.linalg.norm(trajectory[:, 2:4], axis=1),
                label="|V|",
                linewidth=2,
                linestyle="--",
            )
            ax2.legend()

        # Plot 3: Control inputs
        if self.control_history:
            ax3.set_title("Control Inputs")
            ax3.set_xlabel("Time Step")
            ax3.set_ylabel("Control")
            ax3.grid(True, alpha=0.3)

            controls = np.array(self.control_history)
            ax3.plot(controls[:, 0], label="ax (X acceleration)", linewidth=2)
            ax3.plot(controls[:, 1], label="ay (Y acceleration)", linewidth=2)
            ax3.legend()

        # Plot 4: Cost evolution 
        if self.cost_history:
            ax4.set_title("Cost Evolution")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Minimum Cost")
            ax4.grid(True, alpha=0.3)
            ax4.plot(self.cost_history, linewidth=2)
            ax4.set_yscale("log")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "results.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()

    def plot_safety_analysis(self):
        """Plot comprehensive safety analysis showing inflated obstacles and violations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Final trajectory with safety zones
        ax1.set_title(f"Trajectory with Safety Margins (margin = {self.safety_margin})")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax1, show_safety=True)

        # Plot executed trajectory with safety violation highlighting
        if self.history:
            trajectory = np.array(self.history)

            # Plot safe segments in blue
            safe_segments = []
            violation_segments = []

            for i in range(len(trajectory)):
                if i < len(self.safety_violations) and self.safety_violations[i]:
                    violation_segments.append(trajectory[i])
                else:
                    safe_segments.append(trajectory[i])

            if safe_segments:
                safe_traj = np.array(safe_segments)
                ax1.scatter(
                    safe_traj[:, 0],
                    safe_traj[:, 1],
                    c="blue",
                    s=20,
                    alpha=0.8,
                    label="Safe Path",
                    zorder=5,
                )

            if violation_segments:
                violation_traj = np.array(violation_segments)
                ax1.scatter(
                    violation_traj[:, 0],
                    violation_traj[:, 1],
                    c="orange",
                    s=30,
                    alpha=0.9,
                    label="Safety Violations",
                    zorder=6,
                )

            # Draw connecting line
            ax1.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "k-",
                linewidth=2,
                alpha=0.3,
                zorder=1,
            )
            ax1.plot(
                trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start"
            )

        ax1.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")
        ax1.legend()
        ax1.set_aspect("equal")

        # Plot 2: Safety violations over time
        ax2.set_title("Safety Margin Violations Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("In Safety Margin")
        ax2.grid(True, alpha=0.3)

        if self.safety_violations:
            ax2.plot(
                self.safety_violations, "r-", linewidth=2, label="Safety Violation"
            )
            ax2.fill_between(
                range(len(self.safety_violations)),
                self.safety_violations,
                alpha=0.3,
                color="red",
            )

            total_violations = sum(self.safety_violations)
            violation_percentage = total_violations / len(self.safety_violations) * 100
            ax2.text(
                0.02,
                0.98,
                f"Total violations: {total_violations} steps\n"
                f"Percentage: {violation_percentage:.1f}%",
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()

        # Plot 3: Distance to closest obstacle over time
        ax3.set_title("Distance to Closest Obstacle")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Distance")
        ax3.grid(True, alpha=0.3)

        if self.history:
            min_distances_actual = []
            min_distances_inflated = []

            for state in self.history:
                pos = state[:2]
                min_dist_actual = float("inf")
                min_dist_inflated = float("inf")

                for obstacle in self.obstacles:
                    sdf = obstacle.distance_from_surface(pos)
                    min_dist_actual   = min(min_dist_actual,   sdf)
                    min_dist_inflated = min(min_dist_inflated, sdf - obstacle.safety_margin)

                min_distances_actual.append(max(0, min_dist_actual))
                min_distances_inflated.append(min_dist_inflated)

            ax3.plot(
                min_distances_actual,
                "b-",
                linewidth=2,
                label="Distance to Actual Obstacle",
            )
            ax3.plot(
                min_distances_inflated,
                "r-",
                linewidth=2,
                label="Distance to Safety Boundary",
            )
            ax3.axhline(
                y=0, color="r", linestyle="--", alpha=0.7, label="Safety Boundary"
            )
            ax3.axhline(
                y=self.safety_margin,
                color="g",
                linestyle="--",
                alpha=0.7,
                label=f"Safety Margin ({self.safety_margin}m)",
            )
            ax3.legend()

        # Plot 4: Weight concentration over time (from original rollout analysis)
        ax4.set_title("Weight Concentration Over Time")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Effective Sample Size")
        ax4.grid(True, alpha=0.3)

        if self.rollout_history:
            effective_samples = []
            for rollout_data in self.rollout_history:
                weights = rollout_data["weights"]
                eff_samples = 1.0 / np.sum(weights**2)
                effective_samples.append(eff_samples)

            ax4.plot(effective_samples, "b-", linewidth=2)
            ax4.axhline(
                y=self.controller.num_samples / 10,
                color="r",
                linestyle="--",
                label=f"10% of samples ({self.controller.num_samples // 10})",
            )
            ax4.legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "safety_analysis.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()

    def animate_simulation(self, save_animation: bool = False):
        """Create an animation of the simulation (original style)"""
        if not self.history:
            print("No simulation data to animate. Run simulation first.")
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title("MPPI Robot Navigation with Obstacle Avoidance")

            self._draw_all_obstacles(ax, show_safety=False)
            ax.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")

            # Initialize dynamic elements
            (robot,) = ax.plot([], [], "bo", markersize=8, label="Robot")
            (trajectory,) = ax.plot([], [], "b-", alpha=0.6, linewidth=2)

            ax.legend()

            def animate(frame):
                if frame < len(self.history):
                    state = self.history[frame]

                    # Update robot position
                    robot.set_data([state[0]], [state[1]])

                    # Update trajectory
                    trajectory_data = np.array(self.history[: frame + 1])
                    trajectory.set_data(trajectory_data[:, 0], trajectory_data[:, 1])

                return robot, trajectory

            try:
                anim = FuncAnimation(
                    fig,
                    animate,
                    frames=len(self.history),
                    interval=100,
                    blit=True,
                    repeat=True,
                )
            except:
                anim = FuncAnimation(
                    fig,
                    animate,
                    frames=len(self.history),
                    interval=100,
                    blit=False,
                    repeat=True,
                )

            if save_animation:
                try:
                    print("Saving animation... (this may take a while)")
                    save_path = os.path.join(self.output_dir, "mppi_obstacle_avoidance.gif")
                    anim.save(save_path, writer="pillow", fps=10)
                    print(f"Saved: {save_path}")
                except Exception as e:
                    print(f"Failed to save animation: {e}")

            plt.show()
            return anim

        except Exception as e:
            print(f"Animation failed: {e}")
            print("Creating step-by-step visualization instead...")
            self.create_step_by_step_plots()

    def create_step_by_step_plots(self):
        """Create a series of static plots showing the robot's progress (original)"""
        if not self.history:
            print("No simulation data available.")
            return

        trajectory = np.array(self.history)
        n_steps = len(trajectory)

        # Create 6 snapshots of the simulation
        snapshot_indices = np.linspace(0, n_steps - 1, 6, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, step_idx in enumerate(snapshot_indices):
            ax = axes[i]
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Step {step_idx} / {n_steps - 1}")

            self._draw_all_obstacles(ax, show_safety=False)

            # Plot goal
            ax.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")

            # Plot trajectory up to current step
            if step_idx > 0:
                ax.plot(
                    trajectory[: step_idx + 1, 0],
                    trajectory[: step_idx + 1, 1],
                    "b-",
                    alpha=0.6,
                    linewidth=2,
                    label="Path",
                )

            # Plot robot current position
            current_pos = trajectory[step_idx]
            ax.plot(current_pos[0], current_pos[1], "bo", markersize=10, label="Robot")

            # Plot velocity vector if available
            if step_idx < len(self.control_history):
                control = self.control_history[step_idx]
                ax.arrow(
                    current_pos[0],
                    current_pos[1],
                    control[0] * 0.5,
                    control[1] * 0.5,
                    head_width=0.1,
                    head_length=0.1,
                    fc="green",
                    ec="green",
                )

            if i == 0:
                ax.legend()

        plt.tight_layout()
        plt.suptitle(
            "MPPI Robot Navigation - Step by Step Progress", y=1.02, fontsize=16
        )
        save_path = os.path.join(self.output_dir, "step_by_step.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()

    def create_interactive_plot(self):
        """Create an interactive plot that you can step through manually (original)"""
        if not self.history:
            print("No simulation data available.")
            return

        trajectory = np.array(self.history)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax, show_safety=False)

        # Plot goal
        ax.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")

        # Plot full trajectory
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "b-",
            alpha=0.4,
            linewidth=1,
            label="Full Path",
        )

        # Plot start and end positions
        ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=12, label="Start")
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", markersize=12, label="End")

        # Add some key waypoints
        n_waypoints = min(10, len(trajectory))
        waypoint_indices = np.linspace(0, len(trajectory) - 1, n_waypoints, dtype=int)

        for i, idx in enumerate(waypoint_indices):
            ax.plot(
                trajectory[idx, 0], trajectory[idx, 1], "ko", markersize=6, alpha=0.7
            )
            if i % 2 == 0:  # Label every other waypoint to avoid clutter
                ax.annotate(
                    f"t={idx}",
                    (trajectory[idx, 0], trajectory[idx, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.legend()
        ax.set_title("MPPI Robot Navigation - Complete Trajectory")
        save_path = os.path.join(self.output_dir, "interactive_trajectory.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()

    def create_rollout_animation(
        self, save_animation: bool = False, max_rollouts: int = 50
    ):
        """
        Create an animation showing how rollouts evolve over time with safety zones
        """
        if not self.rollout_history:
            print("No rollout data available.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax, show_safety=True)
        ax.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")

        # Initialize dynamic elements
        rollout_lines = []
        for i in range(max_rollouts):
            (line,) = ax.plot([], [], alpha=0.4, linewidth=1)
            rollout_lines.append(line)

        (robot,) = ax.plot(
            [],
            [],
            "ko",
            markersize=12,
            markerfacecolor="white",
            markeredgewidth=3,
            label="Robot",
        )
        (executed_path,) = ax.plot(
            [], [], "g-", linewidth=4, alpha=0.8, label="Executed Path"
        )

        ax.legend()
        ax.set_title("MPPI Rollout Evolution with Safety Margins")

        def animate(frame):
            if frame < len(self.rollout_history):
                rollout_data = self.rollout_history[frame]
                trajectories = rollout_data["trajectories"]
                weights = rollout_data["weights"]
                current_state = rollout_data["current_state"]

                # Sort by weight
                sorted_indices = np.argsort(weights)
                if len(sorted_indices) > max_rollouts:
                    step = len(sorted_indices) // max_rollouts
                    sorted_indices = sorted_indices[::step]

                # Update rollout lines
                colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_indices)))

                for i in range(max_rollouts):
                    if i < len(sorted_indices):
                        idx = sorted_indices[i]
                        traj = trajectories[idx]
                        weight = weights[idx]
                        alpha = 0.2 + 0.6 * (weight / np.max(weights))

                        rollout_lines[i].set_data(traj[:, 0], traj[:, 1])
                        rollout_lines[i].set_color(colors[i])
                        rollout_lines[i].set_alpha(alpha)
                    else:
                        rollout_lines[i].set_data([], [])

                # Update robot position
                robot.set_data([current_state[0]], [current_state[1]])

                # Update executed path
                if frame > 0:
                    executed_traj = np.array(self.history[:frame])
                    executed_path.set_data(executed_traj[:, 0], executed_traj[:, 1])

                ax.set_title(
                    f"MPPI Rollout Evolution with Safety Margins - Timestep {frame}"
                )

            return rollout_lines + [robot, executed_path]

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(self.rollout_history),
            interval=200,
            blit=False,
            repeat=True,
        )

        if save_animation:
            try:
                print("Saving rollout animation...")
                save_path = os.path.join(self.output_dir, "mppi_rollouts_safety.gif")
                anim.save(save_path, writer="pillow", fps=5)
                print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")

        plt.show()
        return anim

    def create_safety_comparison_plot(self):
        """Create a comparison plot showing path with and without safety margins"""
        if not self.history:
            print("No simulation data available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Left plot: Without safety margins (original obstacles only)
        ax1.set_title("Path Planning - Original Obstacles Only")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax1, show_safety=False)

        if self.history:
            trajectory = np.array(self.history)
            ax1.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "b-",
                linewidth=2,
                label="Executed Path",
            )
            ax1.plot(
                trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start"
            )

        ax1.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")
        ax1.legend()
        ax1.set_aspect("equal")

        # Right plot: With safety margins (inflated obstacles)
        ax2.set_title(f"Path Planning - With Safety Margins ({self.safety_margin}m)")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax2, show_safety=True)

        if self.history:
            trajectory = np.array(self.history)

            # Color-code the path based on safety violations
            safe_points = []
            violation_points = []

            for i, state in enumerate(trajectory):
                if i < len(self.safety_violations) and self.safety_violations[i]:
                    violation_points.append(state)
                else:
                    safe_points.append(state)

            # Plot path segments
            ax2.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "k-",
                linewidth=1,
                alpha=0.3,
                zorder=1,
            )

            if safe_points:
                safe_traj = np.array(safe_points)
                ax2.scatter(
                    safe_traj[:, 0],
                    safe_traj[:, 1],
                    c="blue",
                    s=15,
                    alpha=0.8,
                    label="Safe Path",
                    zorder=5,
                )

            if violation_points:
                violation_traj = np.array(violation_points)
                ax2.scatter(
                    violation_traj[:, 0],
                    violation_traj[:, 1],
                    c="orange",
                    s=25,
                    alpha=0.9,
                    label="Safety Violations",
                    zorder=6,
                )

            ax2.plot(
                trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start"
            )

        ax2.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")
        ax2.legend()
        ax2.set_aspect("equal")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "safety_comparison.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()

    def create_detailed_safety_analysis(self):
        """Create detailed analysis of safety performance"""
        if not self.history:
            print("No simulation data available.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Safety violations heatmap over trajectory
        ax1.set_title("Safety Violation Heatmap")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        self._draw_all_obstacles(ax1, show_safety=True)

        if self.history:
            trajectory = np.array(self.history)

            # Create a heatmap showing where violations occurred
            violation_intensities = []
            for i, state in enumerate(trajectory):
                pos = state[:2]
                min_safety_distance = float("inf")

                for obstacle in self.obstacles:
                    safety_dist = obstacle.distance_from_surface(pos) - obstacle.safety_margin
                    min_safety_distance = min(min_safety_distance, safety_dist)

                # Violation intensity: how deep into safety zone
                if min_safety_distance < 0:
                    violation_intensities.append(-min_safety_distance)
                else:
                    violation_intensities.append(0)

            # Plot trajectory colored by violation intensity
            scatter = ax1.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                c=violation_intensities,
                cmap="YlOrRd",
                s=30,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )
            plt.colorbar(scatter, ax=ax1, label="Safety Violation Depth (m)")

        ax1.plot(self.goal[0], self.goal[1], "r*", markersize=15, label="Goal")
        ax1.set_aspect("equal")

        # Plot 2: Safety margin effectiveness
        ax2.set_title("Safety Margin Effectiveness")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Distance (m)")
        ax2.grid(True, alpha=0.3)

        if self.history:
            min_clearances = []
            safety_boundaries = []

            for state in self.history:
                pos = state[:2]
                min_clearance = float("inf")
                min_safety_boundary = float("inf")

                for obstacle in self.obstacles:
                    sdf = obstacle.distance_from_surface(pos)
                    safety_boundary_dist = sdf - obstacle.safety_margin
                    min_clearance = min(min_clearance, sdf)
                    min_safety_boundary = min(min_safety_boundary, safety_boundary_dist)

                min_clearances.append(max(0, min_clearance))
                safety_boundaries.append(min_safety_boundary)

            ax2.plot(
                min_clearances, "b-", linewidth=2, label="Clearance from Obstacles"
            )
            ax2.plot(
                safety_boundaries,
                "r-",
                linewidth=2,
                label="Distance to Safety Boundary",
            )
            ax2.axhline(
                y=0, color="r", linestyle="--", alpha=0.7, label="Safety Boundary"
            )
            ax2.axhline(
                y=self.safety_margin,
                color="g",
                linestyle="--",
                alpha=0.7,
                label=f"Target Safety Margin ({self.safety_margin}m)",
            )

            # Highlight violation regions
            violation_regions = np.array(safety_boundaries) < 0
            if np.any(violation_regions):
                ax2.fill_between(
                    range(len(safety_boundaries)),
                    np.minimum(safety_boundaries, 0),
                    0,
                    where=violation_regions,
                    alpha=0.3,
                    color="red",
                    label="Safety Violations",
                )

            ax2.legend()

        # Plot 3: Obstacle proximity analysis
        ax3.set_title("Proximity to Each Obstacle")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Distance (m)")
        ax3.grid(True, alpha=0.3)

        if self.history:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.obstacles)))

            for i, obstacle in enumerate(self.obstacles):
                distances = []
                for state in self.history:
                    dist = obstacle.distance_from_surface(state[:2])
                    distances.append(max(0.0, dist))

                size_label = (f"r={obstacle.radius:.1f}" if obstacle.shape == "circle"
                              else f"w={obstacle.width:.1f},h={obstacle.height:.1f}")
                ax3.plot(
                    distances,
                    color=colors[i],
                    linewidth=2,
                    label=f"Obstacle {i + 1} [{obstacle.shape}] ({size_label})",
                )

                # Show safety margin for this obstacle
                ax3.axhline(
                    y=obstacle.safety_margin, color=colors[i], linestyle="--", alpha=0.5
                )

            ax3.legend()

        # Plot 4: Performance metrics over time
        ax4.set_title("Performance Metrics")
        ax4.set_xlabel("Time Step")
        ax4.grid(True, alpha=0.3)

        if self.history and self.cost_history:
            # Normalize metrics for comparison
            ax4_twin = ax4.twinx()

            # Distance to goal
            goal_distances = []
            for state in self.history:
                goal_dist = np.linalg.norm(state[:2] - self.goal)
                goal_distances.append(goal_dist)

            ax4.plot(goal_distances, "g-", linewidth=2, label="Distance to Goal")
            ax4.set_ylabel("Distance to Goal (m)", color="g")
            ax4.tick_params(axis="y", labelcolor="g")

            # Cost evolution on secondary axis
            ax4_twin.plot(self.cost_history, "b-", linewidth=2, label="Cost")
            ax4_twin.set_ylabel("Cost", color="b")
            ax4_twin.tick_params(axis="y", labelcolor="b")
            ax4_twin.set_yscale("log")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "detailed_safety_analysis.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.show()


def main():
    """Main function demonstrating all visualization capabilities."""
    parser = argparse.ArgumentParser(
        description="MPPI Controller with random obstacle generation"
    )
    parser.add_argument(
        "-n", "--num-obstacles",
        type=int,
        default=None,
        metavar="N",
        help="Number of obstacles to place (default: random 3-8 each run)",
    )
    parser.add_argument(
        "-s", "--shape",
        choices=["circle", "rectangle", "mixed"],
        default="mixed",
        help="Obstacle shape: circle, rectangle, or mixed (default: mixed)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.2,
        metavar="M",
        help="Safety margin in metres added to every obstacle (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible layouts (default: different each run)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum simulation steps (default: 150)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save plots and animations (default: output)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    print("=" * 60)
    print("MPPI Controller with Comprehensive Visualization")
    print("Including Original Plots + Enhanced Safety Analysis")
    print("=" * 60)
    print(f"Obstacles : {args.num_obstacles if args.num_obstacles else 'random 3-8'}")
    print(f"Shape     : {args.shape}")
    print(f"Safety    : {args.safety_margin} m")

    sim = MPPISimulation(
        safety_margin=args.safety_margin,
        output_dir=args.output_dir,
        num_obstacles=args.num_obstacles,
        obstacle_shape=args.shape,
    )
    print(f"Plots and animations will be saved to: {os.path.abspath(sim.output_dir)}")

    print(f"\nRunning simulation with safety margin: {args.safety_margin}")
    start_time = time.time()
    success = sim.run_simulation(max_steps=args.max_steps)
    end_time = time.time()

    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {success}")

    if sim.history:
        final_state = sim.history[-1]
        final_distance = np.linalg.norm(final_state[:2] - sim.goal)
        print(f"Final distance to goal: {final_distance:.3f}")

        min_clearance = min(
            obstacle.distance_from_surface(state[:2])
            for state in sim.history
            for obstacle in sim.obstacles
        )
        print(f"Minimum clearance from obstacles: {min_clearance:.3f}")
        print(
            f"Safety violations: {sum(sim.safety_violations)} out of {len(sim.safety_violations)} steps"
        )

    print("\n" + "=" * 50)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 50)

    # 1. Original plots 
    print("\n1. Original MPPI Results (4-panel plot)")
    sim.plot_results()

    # 2. Enhanced safety analysis
    print("\n2. Safety Analysis with Inflated Obstacles")
    sim.plot_safety_analysis()

    # 3. Rollout visualization at key timesteps
    print("\n3. Rollout Analysis at Key Timesteps")
    if len(sim.rollout_history) > 0:
        timesteps_to_show = [
            0,
            len(sim.rollout_history) // 2,
            len(sim.rollout_history) - 1,
        ]

        for i, timestep in enumerate(timesteps_to_show):
            if timestep < len(sim.rollout_history):
                print(f"   Showing rollouts at timestep {timestep}")
                sim.plot_rollouts_at_timestep(timestep, max_rollouts=50)

    # 4. Original animation
    print("\n4. Original Simulation Animation")
    try:
        anim1 = sim.animate_simulation(save_animation=True)
    except Exception as e:
        print(f"Original animation failed: {e}")

    # 5. Enhanced rollout animation with safety zones
    print("\n5. Rollout Evolution Animation with Safety Zones")
    try:
        anim2 = sim.create_rollout_animation(save_animation=True, max_rollouts=30)
    except Exception as e:
        print(f"Rollout animation failed: {e}")

    # 6. Step-by-step plots (original)
    print("\n6. Step-by-Step Progress Visualization")
    sim.create_step_by_step_plots()

    # 7. Interactive trajectory plot (original)
    print("\n7. Interactive Complete Trajectory Plot")
    sim.create_interactive_plot()

    # 8. Safety comparison plot
    print("\n8. Safety Comparison: Original vs Inflated Obstacles")
    sim.create_safety_comparison_plot()

    # 9. Detailed safety analysis
    print("\n9. Detailed Safety Analysis (4-panel)")
    sim.create_detailed_safety_analysis()

    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    if sim.cost_history and sim.history:
        print(f"Initial cost: {sim.cost_history[0]:.2f}")
        print(f"Final cost: {sim.cost_history[-1]:.2f}")
        print(
            f"Cost reduction: {(sim.cost_history[0] - sim.cost_history[-1]) / sim.cost_history[0] * 100:.1f}%"
        )
        print(
            f"Total path length: {sum(np.linalg.norm(np.diff(np.array(sim.history)[:, :2], axis=0), axis=1)):.3f}"
        )
        print(
            f"Safety violation rate: {sum(sim.safety_violations) / len(sim.safety_violations) * 100:.1f}%"
        )

    print("\n" + "=" * 50)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
