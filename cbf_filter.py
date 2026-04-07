"""
cbf_filter.py — CBF-QP safety filter for the Go2 MPPI planner.

Theory
------
For each obstacle i, define the barrier function

    h_i(x) = distance_from_surface(pos)   [the obstacle SDF]

The safe set is  S_i = { x : h_i(x) >= 0 }.

The CBF condition (with class-K linear function α(h) = γ·h) is:

    ḣ_i(x, u)  +  γ · h_i(x)  ≥  0

which guarantees forward invariance of S_i under the Go2 unicycle model.

For the unicycle  [ẋ, ẏ] = R(yaw)·[cmd_vx, cmd_vy]:

    ḣ_i = ∇h_i · R(yaw) · [cmd_vx, cmd_vy]

This is *linear* in [cmd_vx, cmd_vy], so the safety filter is a convex QP:

    min_{u}   ½ ‖u − u_nom‖²
    s.t.      ∇h_i · R(yaw) · u[:2]  +  γ · h_i  ≥  0    ∀ i
              u  ∈  box bounds

cmd_yaw does not appear in the instantaneous position rate and is therefore
unconstrained by the CBF (it is still bounded by its box limit).

Gradient computation (Option 1 — analytical piecewise)
-------------------------------------------------------
Circle:
    ∇h = (pos − center) / ‖pos − center‖

Axis-aligned rectangle (box SDF):
    Three regions with distinct closed-form gradients:
      • Corner region (outside, nearest point is a corner):
            ∇h = normalize([qx · sign(dx),  qy · sign(dy)])
      • Left/right face dominates:
            ∇h = [sign(dx), 0]
      • Top/bottom face dominates:
            ∇h = [0, sign(dy)]

    The gradient is discontinuous only *at* geometric corners and edges —
    a measure-zero set never reached by a continuous trajectory.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List

# Match the velocity limits in go2_sim.py / mppi_go2.py
_VX_LIMIT  = 1.5    # m/s  forward / backward
_VY_LIMIT  = 0.8    # m/s  lateral
_YAW_LIMIT = 1.0    # rad/s


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def _sdf_gradient(pos: np.ndarray, obs) -> np.ndarray:
    """
    Analytical piecewise gradient of obs.distance_from_surface w.r.t. pos.

    Returns a unit-length 2-D vector pointing *away* from the nearest point
    on the obstacle surface (i.e., the outward surface normal at the closest
    surface element).

    Parameters
    ----------
    pos : (2,) robot position in world frame [x, y]
    obs : Obstacle instance (circle or rectangle)
    """
    if obs.shape == "circle":
        diff = pos - np.array([obs.x, obs.y])
        n = np.linalg.norm(diff)
        return diff / n if n > 1e-8 else np.array([1.0, 0.0])

    # Axis-aligned rectangle ------------------------------------------------
    hw      = (obs.width  if obs.width  is not None else obs.radius * 2) / 2.0
    hh_rect = (obs.height if obs.height is not None else obs.radius * 2) / 2.0

    dx = pos[0] - obs.x
    dy = pos[1] - obs.y
    qx = abs(dx) - hw       # signed excess beyond half-width
    qy = abs(dy) - hh_rect  # signed excess beyond half-height

    if qx > 0 and qy > 0:
        # Corner region — robot is outside, nearest surface point is a corner.
        # Gradient points from the corner toward the robot (outward direction).
        n = np.hypot(qx, qy) + 1e-8
        return np.array([qx * np.sign(dx), qy * np.sign(dy)]) / n

    if qx >= qy:
        # Left / right face dominates (inside OR outside with x dominant)
        return np.array([np.sign(dx), 0.0])

    # Top / bottom face dominates
    return np.array([0.0, np.sign(dy)])


# ---------------------------------------------------------------------------
# CBF safety filter
# ---------------------------------------------------------------------------

class CBFSafetyFilter:
    """
    CBF-QP safety filter for Go2 body-frame velocity commands.

    State  : [x, y, yaw, vx_world, vy_world]
    Control: [cmd_vx, cmd_vy, cmd_yaw]   (body frame)

    Handles both circle and rectangle obstacles via piecewise analytical
    SDF gradients.  The resulting QP has 3 decision variables and one linear
    inequality constraint per obstacle, so it solves in < 1 ms for typical
    environments (3–10 obstacles).
    """

    def __init__(self, gamma: float = 1.0):
        """
        Parameters
        ----------
        gamma : CBF decay-rate coefficient (slope of the class-K function).
                Larger γ makes the filter activate sooner / more aggressively.
                Typical range: 0.5 – 2.0.
        """
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        state: np.ndarray,
        u_nominal: np.ndarray,
        obstacles: List,
    ) -> np.ndarray:
        """
        Project u_nominal onto the CBF-safe control set.

        Parameters
        ----------
        state     : [x, y, yaw, vx_world, vy_world]
        u_nominal : [cmd_vx, cmd_vy, cmd_yaw] from MPPI (body frame)
        obstacles : list of Obstacle (any mix of circle / rectangle)

        Returns
        -------
        u_safe : [cmd_vx, cmd_vy, cmd_yaw] — nearest control satisfying all
                 CBF constraints  ḣ_i + γ·h_i ≥ 0  and box bounds.
        """
        pos   = state[:2]
        yaw   = float(state[2])
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        A_rows: List[np.ndarray] = []
        b_vals: List[float]      = []

        for obs in obstacles:
            h    = float(obs.distance_from_surface(pos))
            grad = _sdf_gradient(pos, obs)   # (2,) outward unit normal

            # ḣ = ∇h · R(yaw) · [cmd_vx, cmd_vy]
            #
            # R(yaw) maps body frame to world frame:
            #   body-x axis in world = [ cos_y,  sin_y ]
            #   body-y axis in world = [-sin_y,  cos_y ]
            #
            # So  ḣ = (grad · body-x) · cmd_vx  +  (grad · body-y) · cmd_vy
            a_vx = grad[0] * cos_y + grad[1] * sin_y
            a_vy = -grad[0] * sin_y + grad[1] * cos_y

            A_rows.append(np.array([a_vx, a_vy]))
            #b_vals.append(self.gamma * h)
            b_vals.append(self.gamma * (h - obs.safety_margin))

        if not A_rows:
            return u_nominal.copy()

        A = np.array(A_rows)    # (N_obs, 2) — constraint Jacobians
        b = np.array(b_vals)    # (N_obs,)   — slack terms  γ · h_i

        return self._solve_qp(u_nominal, A, b)

    # ------------------------------------------------------------------
    # Internal QP solver
    # ------------------------------------------------------------------

    def _solve_qp(
        self,
        u_ref: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the CBF-QP with SLSQP (scipy).

        Decision variables : u = [cmd_vx, cmd_vy, cmd_yaw]  (3-D)
        Objective          : ½ ‖u − u_ref‖²  (minimum-norm modification)
        CBF constraints    : A[i] @ u[:2] + b[i] ≥ 0   for each obstacle i
        Box constraints    : u ∈ [−limits, +limits]

        Falls back to zero translation (robot stops) when infeasible,
        which is safe because a stationary robot cannot move into an obstacle.
        """
        bounds = [
            (-_VX_LIMIT,  _VX_LIMIT),
            (-_VY_LIMIT,  _VY_LIMIT),
            (-_YAW_LIMIT, _YAW_LIMIT),
        ]

        # One inequality constraint per obstacle: A[i] @ u[:2] + b[i] >= 0
        cbf_ineq = [
            {
                'type': 'ineq',
                'fun': lambda u, i=i: float(A[i] @ u[:2]) + b[i],
                'jac': lambda u, i=i: np.array([A[i, 0], A[i, 1], 0.0]),
            }
            for i in range(len(b))
        ]

        res = minimize(
            lambda u: 0.5 * float((u - u_ref) @ (u - u_ref)),
            u_ref.copy(),
            jac=lambda u: (u - u_ref),
            method='SLSQP',
            bounds=bounds,
            constraints=cbf_ineq,
            options={'ftol': 1e-9, 'maxiter': 300, 'disp': False},
        )

        # status == 0 : optimal solution found
        # status == 9 : iteration limit reached — take best found so far
        if res.success or res.status == 9:
            return res.x.astype(np.float64)

        # Infeasible (very rare — robot already inside an obstacle).
        # Safest action: stop translating, keep current yaw rate.
        print(
            f"CBF-QP infeasible (scipy status={res.status}): "
            "commanding zero translation"
        )
        return np.array([0.0, 0.0, float(u_ref[2])])
