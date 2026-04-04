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


# Physical limits of the Go2 (from walk-these-ways training config)
VX_LIMIT = 1.5    # m/s forward/back
VY_LIMIT = 0.8    # m/s lateral
YAW_LIMIT = 1.0   # rad/s


class MPPIGo2Controller(MPPIController):
    """
    MPPI high-level planner for the Go2.

    State:   [x, y, yaw, vx_world, vy_world]
    Control: [cmd_vx, cmd_vy, cmd_yaw_rate]  (body frame)
    """

    def __init__(
        self,
        horizon: int = 20,
        num_samples: int = 500,
        dt: float = 0.1,
        lambda_: float = 1.0,
        sigma: float = 0.5,
        control_bounds: Tuple[float, float] = (-1.5, 1.5),
        safety_margin: float = 0.5,
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
        self.Q_velocity = 0.2

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

            # Control effort
            cost += self.R * np.linalg.norm(control) ** 2

            # Velocity magnitude penalty (world-frame vx, vy)
            velocity = state[3:5]
            cost += self.Q_velocity * np.linalg.norm(velocity) ** 2

            # Obstacle costs (signed-distance based)
            pos = state[:2]
            for obstacle in obstacles:
                dist = obstacle.distance_from_surface(pos)

                if dist < obstacle.safety_margin:
                    penetration = obstacle.safety_margin - dist
                    cost += self.Q_obstacle * (penetration ** 2)

                    if dist < 0.1:
                        collision_penalty += 2000.0 * np.exp(-10 * max(dist, 0.0))

                if dist < 0.0:
                    collision_penalty += 10000.0

        cost += collision_penalty
        return cost
