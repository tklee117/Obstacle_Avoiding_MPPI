"""
MPPI controller adapted for the Unitree Go2 high-level navigation.

Changes from mppi_hard_constraint.py:
  - state_dim=5: [x, y, yaw, vx, vy]
  - control_dim=3: [cmd_vx, cmd_vy, cmd_yaw_rate]  (body-frame velocity commands)
  - dynamics: unicycle model instead of point-mass double integrator
  - compute_cost: velocity cost uses world-frame velocity components state[3:5]

Everything else (sampling, weighting, warm-start, obstacle SDF) is unchanged.
"""

import numpy as np
from typing import List, Tuple

from mppi_hard_constraint import MPPIController, Obstacle
from cbf_filter import CBFSafetyFilter


# Physical limits of the Go2 (from walk-these-ways training config)
VX_LIMIT = 1.5    # m/s forward/back
VY_LIMIT = 0.8    # m/s lateral
YAW_LIMIT = 1.0   # rad/s

# Go2 body half-extents for rectangular footprint obstacle cost
GO2_HALF_LENGTH = 0.35   # metres fore–aft  (body ≈ 0.70 m long)
GO2_HALF_WIDTH  = 0.15   # metres lateral   (body ≈ 0.30 m wide)


class MPPIGo2Controller(MPPIController):
    """
    MPPI high-level planner for the Go2.

    State:   [x, y, yaw, vx_world, vy_world]
    Control: [cmd_vx, cmd_vy, cmd_yaw_rate]  (body frame)
    """

    def __init__(
        self,
        horizon: int = 30,
        num_samples: int = 500,
        dt: float = 0.1,
        lambda_: float = 7.0,
        sigma: float = 0.9,
        control_bounds: Tuple[float, float] = (-1.5, 1.5),
        safety_margin: float = 0.5,
        cbf_gamma: float = 1.0,
    ):
        super().__init__(
            horizon=horizon,
            num_samples=num_samples,
            control_dim=3,
            state_dim=5,
            dt=dt,
            lambda_=lambda_,
            sigma=sigma,
            control_bounds=control_bounds,
            safety_margin=safety_margin,
        )
        # Tighter velocity regulation for the legged robot
        self.Q_velocity = 0.05
        self.Q_goal_running = 1.0
        # Heading-toward-goal cost weight; only active when dist > 0.3 m
        self.Q_yaw = 1.5

        # CBF safety filter — post-processes MPPI output with a hard guarantee
        self.cbf = CBFSafetyFilter(gamma=cbf_gamma)

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Unicycle model for Go2 high-level planning.

        State:   [x, y, yaw, vx_world, vy_world]
        Control: [cmd_vx, cmd_vy, cmd_yaw_rate]  (body frame)
        """
        x, y, yaw, _, _ = state
        cmd_vx, cmd_vy, cmd_yaw = control

        # Clip to Go2 physical limits
        cmd_vx  = np.clip(cmd_vx,  -VX_LIMIT,  VX_LIMIT)
        cmd_vy  = np.clip(cmd_vy,  -VY_LIMIT,  VY_LIMIT)
        cmd_yaw = np.clip(cmd_yaw, -YAW_LIMIT, YAW_LIMIT)

        # Rotate body-frame velocity to world frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        world_vx = cmd_vx * cos_yaw - cmd_vy * sin_yaw
        world_vy = cmd_vx * sin_yaw + cmd_vy * cos_yaw

        return np.array([
            x   + world_vx * self.dt,
            y   + world_vy * self.dt,
            yaw + cmd_yaw  * self.dt,
            world_vx,
            world_vy,
        ])

    def compute_cost(
        self,
        trajectory: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        control_sequence: np.ndarray,
    ) -> float:
        """
        Same cost structure as the base class, but velocity cost uses
        world-frame velocity state[3:5] instead of state[2:4].
        """
        cost = 0.0
        collision_penalty = 0.0

        # Terminal goal cost (position only)
        final_pos = trajectory[-1, :2]
        cost += self.Q_goal * np.linalg.norm(final_pos - goal) ** 2

        for t in range(self.horizon):
            state   = trajectory[t]
            control = control_sequence[t]
            x, y, yaw = state[0], state[1], state[2]

            # Running goal-progress cost — breaks local minima in front of walls
            cost += self.Q_goal_running * np.linalg.norm(state[:2] - goal) ** 2

            # Heading-toward-goal cost — drives yaw to face the current carrot.
            # Only active when far enough away that atan2 is well-conditioned.
            dist_to_goal = np.hypot(goal[0] - x, goal[1] - y)
            if dist_to_goal > 0.3:
                desired_yaw = np.arctan2(goal[1] - y, goal[0] - x)
                yaw_err = (yaw - desired_yaw + np.pi) % (2 * np.pi) - np.pi
                cost += self.Q_yaw * yaw_err ** 2

            # Control effort — translational only; yaw is guided by Q_yaw above
            # so excluding cmd_yaw here lets MPPI rotate freely without double-
            # penalising heading corrections.
            cost += self.R * (control[0] ** 2 + control[1] ** 2)

            # Velocity magnitude penalty (world-frame vx, vy)
            velocity = state[3:5]
            cost += self.Q_velocity * np.linalg.norm(velocity) ** 2

            # Obstacle costs — bounding-rectangle footprint.
            # Evaluate the SDF at each of the 4 corners of the robot's body
            # rectangle and take the minimum, giving the true closest clearance
            # from any part of the dog's body to each obstacle surface.
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            # 4 corners in body frame (long-axis = x, lateral = y)
            corners = []
            for lx, ly in (( GO2_HALF_LENGTH,  GO2_HALF_WIDTH),
                            ( GO2_HALF_LENGTH, -GO2_HALF_WIDTH),
                            (-GO2_HALF_LENGTH,  GO2_HALF_WIDTH),
                            (-GO2_HALF_LENGTH, -GO2_HALF_WIDTH)):
                corners.append(np.array([
                    x + lx * cos_y - ly * sin_y,
                    y + lx * sin_y + ly * cos_y,
                ]))

            for obstacle in obstacles:
                # Minimum clearance from the dog's footprint to this obstacle
                dist = min(obstacle.distance_from_surface(c) for c in corners)

                if dist < obstacle.safety_margin:
                    penetration = obstacle.safety_margin - dist
                    cost += self.Q_obstacle * (penetration ** 2)

                    if dist < 0.1:
                        collision_penalty += 2000.0 * np.exp(-10 * max(dist, 0.0))

                if dist < 0.0:
                    collision_penalty += 10000.0

        cost += collision_penalty
        return cost

    def apply_cbf_filter(
        self,
        state: np.ndarray,
        u_nominal: np.ndarray,
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        """
        Apply the CBF-QP safety filter to the MPPI-optimal control.

        Finds the control u* closest to u_nominal that satisfies:

            ḣ_i(x, u*)  +  γ · h_i(x)  ≥  0    for every obstacle i

        where  h_i = distance_from_surface(pos)  (the obstacle SDF).
        The gradient ∇h_i is computed analytically for both circle and
        rectangle shapes (piecewise, Option 1).

        Parameters
        ----------
        state     : [x, y, yaw, vx_world, vy_world]
        u_nominal : [cmd_vx, cmd_vy, cmd_yaw] from MPPI
        obstacles : list of Obstacle (any mix of circle / rectangle)

        Returns
        -------
        u_safe : CBF-filtered control with the same shape as u_nominal.
                 Equals u_nominal when all CBF constraints are already
                 satisfied; otherwise the minimum-norm modification.
        """
        return self.cbf.filter(state, u_nominal, obstacles)
