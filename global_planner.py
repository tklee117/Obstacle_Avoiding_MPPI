"""
global_planner.py — Grid-based A* planner for MPPI waypoint guidance.

Builds a 2-D occupancy grid from obstacle SDFs and finds a collision-free
path from start to goal.  MPPI then tracks a "carrot" point on this path
(pure-pursuit style) rather than the raw goal, breaking local minima in
front of walls.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional

from mppi_hard_constraint import Obstacle


class AStarPlanner:
    """
    2-D grid A* planner backed by obstacle signed-distance fields.

    Usage
    -----
        planner = AStarPlanner(obstacles, bounds=(-1, 8, -1, 8))
        path    = planner.plan(start, goal)               # list[np.ndarray]
        carrot  = planner.get_carrot(path, robot_pos, lookahead=1.5)
    """

    def __init__(
        self,
        obstacles: List[Obstacle],
        bounds: Tuple[float, float, float, float],  # xmin, xmax, ymin, ymax
        resolution: float = 0.15,
        safety_margin: float = 0.5,
    ):
        self.obstacles     = obstacles
        self.resolution    = resolution
        self.safety_margin = safety_margin

        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.nx = int(np.ceil((self.xmax - self.xmin) / resolution)) + 1
        self.ny = int(np.ceil((self.ymax - self.ymin) / resolution)) + 1

        print(f"A* grid: {self.nx}×{self.ny} cells at {resolution:.2f} m resolution")
        self.grid = self._build_grid()
        free = int(np.sum(~self.grid))
        print(f"  Free cells: {free}/{self.nx * self.ny}")

    # ── grid helpers ──────────────────────────────────────────────────────────

    def _build_grid(self) -> np.ndarray:
        """True = occupied (inside inflated obstacle), False = free."""
        grid = np.zeros((self.nx, self.ny), dtype=bool)
        for ix in range(self.nx):
            for iy in range(self.ny):
                pos = self._idx_to_pos(ix, iy)
                for obs in self.obstacles:
                    if obs.distance_from_surface(pos) < self.safety_margin:
                        grid[ix, iy] = True
                        break
        return grid

    def _pos_to_idx(self, pos: np.ndarray) -> Tuple[int, int]:
        ix = int(round((pos[0] - self.xmin) / self.resolution))
        iy = int(round((pos[1] - self.ymin) / self.resolution))
        return (
            int(np.clip(ix, 0, self.nx - 1)),
            int(np.clip(iy, 0, self.ny - 1)),
        )

    def _idx_to_pos(self, ix: int, iy: int) -> np.ndarray:
        return np.array([
            self.xmin + ix * self.resolution,
            self.ymin + iy * self.resolution,
        ])

    def _nearest_free(
        self, idx: Tuple[int, int], search_radius: int = 8
    ) -> Optional[Tuple[int, int]]:
        """Spiral outward from idx to find the nearest free cell."""
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nb = (idx[0] + dx, idx[1] + dy)
                    if (0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny
                            and not self.grid[nb]):
                        return nb
        return None

    # ── A* ────────────────────────────────────────────────────────────────────

    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """
        Run A* from start to goal.

        Returns a list of (x, y) waypoints (including start and goal),
        or None if no path exists.
        """
        s = self._pos_to_idx(start)
        g = self._pos_to_idx(goal)

        if self.grid[g]:
            print("A* warning: goal cell is occupied — searching for nearest free cell")
            g = self._nearest_free(g)
            if g is None:
                print("A*: no free cell near goal — aborting")
                return None

        if self.grid[s]:
            print("A* warning: start cell is occupied — searching for nearest free cell")
            s = self._nearest_free(s)
            if s is None:
                print("A*: no free cell near start — aborting")
                return None

        # 8-connected neighbours with precomputed step costs
        straight = self.resolution
        diag     = np.sqrt(2) * self.resolution
        neighbours = [
            ( 1,  0, straight), (-1,  0, straight),
            ( 0,  1, straight), ( 0, -1, straight),
            ( 1,  1, diag),     ( 1, -1, diag),
            (-1,  1, diag),     (-1, -1, diag),
        ]

        # (f, g_cost, node)
        open_heap: list = []
        heapq.heappush(open_heap, (0.0, 0.0, s))
        came_from: dict = {}
        g_score: dict   = {s: 0.0}

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)

            if cur == g:
                path = self._reconstruct(came_from, cur)
                print(f"A*: path found with {len(path)} waypoints")
                return path

            # Skip stale heap entries
            if g_cur > g_score.get(cur, float("inf")):
                continue

            for dx, dy, step_cost in neighbours:
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny):
                    continue
                if self.grid[nb]:
                    continue
                tentative = g_cur + step_cost
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = cur
                    g_score[nb]   = tentative
                    h = np.hypot(nb[0] - g[0], nb[1] - g[1]) * self.resolution
                    heapq.heappush(open_heap, (tentative + h, tentative, nb))

        print("A*: no path found")
        return None

    def _reconstruct(self, came_from: dict, current) -> List[np.ndarray]:
        path = []
        while current in came_from:
            path.append(self._idx_to_pos(*current))
            current = came_from[current]
        path.append(self._idx_to_pos(*current))
        path.reverse()
        return path

    # ── carrot-point (pure pursuit) ───────────────────────────────────────────

    def get_carrot(
        self,
        path: List[np.ndarray],
        robot_pos: np.ndarray,
        lookahead: float = 1.5,
    ) -> np.ndarray:
        """
        Return the point on `path` that is `lookahead` metres ahead of
        the robot's nearest point on the path.  Falls back to the final
        waypoint when the lookahead overshoots the end.
        """
        if not path:
            return robot_pos

        # Nearest waypoint
        dists   = [np.linalg.norm(p - robot_pos) for p in path]
        closest = int(np.argmin(dists))

        # Walk forward until lookahead distance is consumed
        accumulated = 0.0
        for i in range(closest, len(path) - 1):
            seg_len = float(np.linalg.norm(path[i + 1] - path[i]))
            if seg_len == 0.0:
                continue
            if accumulated + seg_len >= lookahead:
                remaining = lookahead - accumulated
                direction = (path[i + 1] - path[i]) / seg_len
                return path[i] + remaining * direction
            accumulated += seg_len

        return path[-1]
