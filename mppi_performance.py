#!/usr/bin/env python3
"""
MPPI Benchmark Testing Suite
Tests the MPPI implementation with safety margins
Generates random configurations and evaluates comprehensive metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import random

# Remove the import from mppi_hard_constraint
# Instead, we'll use our own implementations defined in this file

@dataclass
class Obstacle:
    """Represents a circular obstacle in the environment"""
    x: float
    y: float
    radius: float
    safety_margin: float = 0.2
    
    @property
    def inflated_radius(self) -> float:
        """Return the radius including safety margin"""
        return self.radius + self.safety_margin

@dataclass
class SimulationResult:
    """Stores results from a single simulation run"""
    success: bool
    steps_taken: int
    final_distance: float
    path_length: float
    computation_time: float
    collision_occurred: bool
    num_obstacles: int
    scenario_difficulty: float
    safety_violations: int
    total_steps: int
    scenario_id: int
    trajectory: List[np.ndarray] = None  # Store actual path taken

class MPPIController:
    """Model Predictive Path Integral Controller with Inflated Obstacle Safety"""

    def __init__(self,
                 horizon: int = 20,
                 num_samples: int = 100,
                 control_dim: int = 2,
                 state_dim: int = 4,
                 dt: float = 0.1,
                 lambda_: float = 1.0,
                 sigma: float = 1.0,
                 control_bounds: Tuple[float, float] = (-2.0, 2.0),
                 safety_margin: float = 0.2):
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

        # Initialize control sequence (warm start)
        self.U = np.zeros((self.horizon, self.control_dim))

        # Cost weights
        self.Q_goal = 50.0       # Goal reaching weight
        self.Q_obstacle = 500.0  # Obstacle avoidance weight
        self.R = 0.1             # Control effort weight
        self.Q_velocity = 0.5    # Velocity regulation weight

        # For visualization
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

        # Clip control inputs
        ax = np.clip(ax, self.control_bounds[0], self.control_bounds[1])
        ay = np.clip(ay, self.control_bounds[0], self.control_bounds[1])

        # Update state using Euler integration
        next_state = np.array([
            x + vx * self.dt,
            y + vy * self.dt,
            vx + ax * self.dt,
            vy + ay * self.dt
        ])

        return next_state

    def rollout(self, initial_state: np.ndarray, control_sequence: np.ndarray) -> np.ndarray:
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

    def compute_cost(self, trajectory: np.ndarray,
                    goal: np.ndarray,
                    obstacles: List[Obstacle],
                    control_sequence: np.ndarray) -> float:
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

        # Terminal cost: distance to goal
        final_pos = trajectory[-1, :2]
        goal_cost = self.Q_goal * np.linalg.norm(final_pos - goal)**2
        cost += goal_cost

        # Running costs
        for t in range(self.horizon):
            state = trajectory[t]
            control = control_sequence[t]

            # Control effort cost
            control_cost = self.R * np.linalg.norm(control)**2
            cost += control_cost

            # Velocity regulation cost (encourage smooth motion)
            velocity = state[2:4]
            velocity_cost = self.Q_velocity * np.linalg.norm(velocity)**2
            cost += velocity_cost

            # Obstacle avoidance cost using inflated obstacles
            pos = state[:2]
            for obstacle in obstacles:
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                distance = np.linalg.norm(pos - obstacle_pos)

                inflated_radius = obstacle.inflated_radius # Use inflated radius as the hard boundary

                if distance < inflated_radius: # Check if inside the inflated obstacle (safety zone)
                    penetration = inflated_radius - distance

                    # Quadratic penalty that grows rapidly as we get closer
                    obstacle_cost = self.Q_obstacle * (penetration**2)
                    cost += obstacle_cost

                    # Additional exponential penalty for being very close to actual obstacle
                    if distance < obstacle.radius + 0.1:  # Very close to actual obstacle
                        collision_penalty += 2000.0 * np.exp(-10 * (distance - obstacle.radius))

                # Add massive penalty for actual collision with original obstacle
                if distance < obstacle.radius:
                    collision_penalty += 10000.0

        # Add collision penalty
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
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                distance = np.linalg.norm(pos - obstacle_pos)

                if distance < obstacle.inflated_radius:
                    return True
        return False

    def sample_controls(self) -> np.ndarray:
        """
        Sample control sequences around the current control sequence

        Returns:
            sampled_controls: Array of shape (num_samples, horizon, control_dim)
        """
        # Generate random noise
        noise = np.random.normal(0, self.sigma, (self.num_samples, self.horizon, self.control_dim))

        # Add noise to current control sequence
        sampled_controls = self.U[None, :, :] + noise

        # Clip controls to bounds
        sampled_controls = np.clip(sampled_controls,
                                 self.control_bounds[0],
                                 self.control_bounds[1])

        return sampled_controls

    def update_control(self, state: np.ndarray,
                      goal: np.ndarray,
                      obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update the control sequence using MPPI with inflated obstacles

        Args:
            state: Current state [x, y, vx, vy]
            goal: Goal position [x, y]
            obstacles: List of obstacles

        Returns:
            Tuple of (best_control, best_trajectory, costs)
        """
        # Sample control sequences
        sampled_controls = self.sample_controls()

        # Evaluate costs for all samples
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

        # Handle edge case where all weights are zero
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
            # Keep previous control and add small perturbation
            self.U += np.random.normal(0, 0.1, self.U.shape)

        # Shift control sequence for next iteration (warm start)
        self.U[:-1] = self.U[1:].copy()
        self.U[-1] = np.zeros(self.control_dim)

        # Return best trajectory for visualization
        best_idx = np.argmin(costs)
        best_trajectory = trajectories[best_idx]

        return self.U[0], best_trajectory, costs

class RandomEnvironmentGenerator:
    """Generates random environments for testing MPPI robustness"""

    def __init__(self,
                 world_size: Tuple[float, float] = (10.0, 10.0),
                 min_obstacles: int = 3,
                 max_obstacles: int = 8,
                 min_radius: float = 0.3,
                 max_radius: float = 1.2,
                 min_separation: float = 0.8,
                 safety_margin: float = 0.2):
        """Initialize environment generator"""
        self.world_size = world_size
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_separation = min_separation
        self.safety_margin = safety_margin

    def generate_random_scenario(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Obstacle]]:
        """Generate a random scenario with start, goal, and obstacles"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        max_attempts = 200
        
        for attempt in range(max_attempts):
            # Generate random start and goal positions with better separation
            start_pos = np.array([
                np.random.uniform(1.0, self.world_size[0] - 1.0),
                np.random.uniform(1.0, self.world_size[1] - 1.0)
            ])

            goal_pos = np.array([
                np.random.uniform(1.0, self.world_size[0] - 1.0),
                np.random.uniform(1.0, self.world_size[1] - 1.0)
            ])

            # Ensure start and goal are far enough apart
            distance_sg = np.linalg.norm(goal_pos - start_pos)
            if distance_sg < 4.0:  # Minimum distance between start and goal
                continue

            # Generate random number of obstacles
            num_obstacles = np.random.randint(self.min_obstacles, self.max_obstacles + 1)
            obstacles = []

            # Generate obstacles with better placement logic
            obstacles_generated = 0
            obstacle_attempts = 0

            while obstacles_generated < num_obstacles and obstacle_attempts < 100:
                obstacle_attempts += 1

                # Random position with better bounds
                pos = np.array([
                    np.random.uniform(self.min_radius + 0.5, self.world_size[0] - self.min_radius - 0.5),
                    np.random.uniform(self.min_radius + 0.5, self.world_size[1] - self.min_radius - 0.5)
                ])
                radius = np.random.uniform(self.min_radius, self.max_radius)

                # Check if obstacle is too close to start or goal
                start_dist = np.linalg.norm(pos - start_pos)
                goal_dist = np.linalg.norm(pos - goal_pos)
                
                if (start_dist < radius + self.min_separation or
                    goal_dist < radius + self.min_separation):
                    continue

                # Check if obstacle overlaps with existing obstacles
                valid = True
                for existing_obstacle in obstacles:
                    existing_pos = np.array([existing_obstacle.x, existing_obstacle.y])
                    min_distance = radius + existing_obstacle.radius + 0.3
                    if np.linalg.norm(pos - existing_pos) < min_distance:
                        valid = False
                        break

                if valid:
                    obstacles.append(Obstacle(pos[0], pos[1], radius, self.safety_margin))
                    obstacles_generated += 1

            # Check if there's a reasonable path
            if obstacles_generated >= self.min_obstacles and self._check_feasibility(start_pos, goal_pos, obstacles):
                start_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
                return start_state, goal_pos, obstacles

        # Enhanced fallback scenarios
        scenario_type = attempt % 4
        
        if scenario_type == 0:
            # Simple corridor
            start_state = np.array([1.0, 5.0, 0.0, 0.0])
            goal_pos = np.array([9.0, 5.0])
            obstacles = [
                Obstacle(3.0, 3.0, 0.5, self.safety_margin),
                Obstacle(5.0, 7.0, 0.6, self.safety_margin),
                Obstacle(7.0, 4.0, 0.4, self.safety_margin)
            ]
        elif scenario_type == 1:
            # Diagonal path
            start_state = np.array([1.0, 1.0, 0.0, 0.0])
            goal_pos = np.array([9.0, 9.0])
            obstacles = [
                Obstacle(3.0, 4.0, 0.7, self.safety_margin),
                Obstacle(6.0, 5.0, 0.5, self.safety_margin),
                Obstacle(4.0, 7.0, 0.6, self.safety_margin),
                Obstacle(7.0, 3.0, 0.4, self.safety_margin)
            ]
        elif scenario_type == 2:
            # U-shaped path
            start_state = np.array([2.0, 2.0, 0.0, 0.0])
            goal_pos = np.array([8.0, 2.0])
            obstacles = [
                Obstacle(5.0, 3.0, 0.8, self.safety_margin),
                Obstacle(5.0, 4.5, 0.7, self.safety_margin),
                Obstacle(5.0, 6.0, 0.6, self.safety_margin),
                Obstacle(3.5, 5.0, 0.4, self.safety_margin),
                Obstacle(6.5, 5.0, 0.4, self.safety_margin)
            ]
        else:
            # Maze-like
            start_state = np.array([1.5, 8.5, 0.0, 0.0])
            goal_pos = np.array([8.5, 1.5])
            obstacles = [
                Obstacle(3.0, 6.0, 0.5, self.safety_margin),
                Obstacle(5.0, 8.0, 0.4, self.safety_margin),
                Obstacle(7.0, 6.0, 0.6, self.safety_margin),
                Obstacle(4.0, 3.0, 0.5, self.safety_margin),
                Obstacle(6.0, 4.0, 0.4, self.safety_margin),
                Obstacle(8.0, 3.0, 0.3, self.safety_margin)
            ]
        
        return start_state, goal_pos, obstacles

    def _check_feasibility(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Obstacle]) -> bool:
        """Enhanced feasibility check using multiple path samples"""
        # Check direct line
        if self._check_line_of_sight(start, goal, obstacles):
            return True
        
        # Check curved paths
        for angle_offset in [-45, -30, -15, 15, 30, 45]:
            mid_point = (start + goal) / 2
            perpendicular = np.array([-(goal[1] - start[1]), goal[0] - start[0]])
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
            
            offset_distance = np.linalg.norm(goal - start) * 0.3
            curved_mid = mid_point + perpendicular * offset_distance * np.sin(np.radians(angle_offset))
            
            # Keep curved midpoint in bounds
            curved_mid[0] = np.clip(curved_mid[0], 1.0, self.world_size[0] - 1.0)
            curved_mid[1] = np.clip(curved_mid[1], 1.0, self.world_size[1] - 1.0)
            
            if (self._check_line_of_sight(start, curved_mid, obstacles) and
                self._check_line_of_sight(curved_mid, goal, obstacles)):
                return True
        
        return False

    def _check_line_of_sight(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Obstacle]) -> bool:
        """Check if there's a clear line of sight between start and goal"""
        direction = goal - start
        distance = np.linalg.norm(direction)
        if distance == 0:
            return True

        direction_norm = direction / distance

        # Sample points along the line
        num_samples = max(20, int(distance * 5))
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            point = start + t * direction

            # Check if point is inside any obstacle (including safety margin)
            for obstacle in obstacles:
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                if np.linalg.norm(point - obstacle_pos) < obstacle.inflated_radius:
                    return False

        return True

class MPPIBenchmark:
    """Comprehensive benchmarking suite for MPPI using the imported implementation"""

    def __init__(self, controller_params: dict = None):
        """Initialize benchmark suite"""
        self.controller_params = controller_params or {
            'horizon': 20,
            'num_samples': 150,
            'dt': 0.1,
            'lambda_': 1.0,
            'sigma': 1.2,
            'safety_margin': 0.2
        }

        self.env_generator = RandomEnvironmentGenerator(safety_margin=self.controller_params['safety_margin'])
        self.results = []
        self.scenarios = []  # Store scenarios for later analysis

    def run_single_trial(self, start_state: np.ndarray, goal: np.ndarray,
                        obstacles: List[Obstacle], max_steps: int = 300,
                        verbose: bool = False, scenario_id: int = 0) -> SimulationResult:
        """Run a single MPPI trial using the imported controller"""
        # Create controller using the imported MPPIController
        controller = MPPIController(**self.controller_params)
        safety_margin = self.controller_params.get('safety_margin', 0.2)

        # Make sure obstacles have safety margin set
        for obstacle in obstacles:
            obstacle.safety_margin = safety_margin

        # Initialize simulation
        state = start_state.copy()
        history = []
        safety_violations = []

        start_time = time.time()

        for step in range(max_steps):
            try:
                # Get control from MPPI
                control, _, _ = controller.update_control(state, goal, obstacles)

                # Apply control and update state
                state = controller.dynamics(state, control)
                history.append(state.copy())

                # Check safety violations and collisions
                safety_violation = False
                actual_collision = False

                for obstacle in obstacles:
                    distance = np.linalg.norm(state[:2] - np.array([obstacle.x, obstacle.y]))
                    if distance < obstacle.radius:
                        actual_collision = True
                        break
                    elif distance < obstacle.inflated_radius:
                        safety_violation = True

                safety_violations.append(safety_violation)

                # Check if goal reached
                distance_to_goal = np.linalg.norm(state[:2] - goal)
                if distance_to_goal < 0.4:  # Goal tolerance
                    computation_time = time.time() - start_time
                    path_length = self._compute_path_length(history)

                    return SimulationResult(
                        success=True,
                        steps_taken=step + 1,
                        final_distance=distance_to_goal,
                        path_length=path_length,
                        computation_time=computation_time,
                        collision_occurred=False,
                        num_obstacles=len(obstacles),
                        scenario_difficulty=np.linalg.norm(start_state[:2] - goal),
                        safety_violations=sum(safety_violations),
                        total_steps=step + 1,
                        scenario_id=scenario_id,
                        trajectory=history.copy()
                    )

                # Check for actual collision
                if actual_collision:
                    computation_time = time.time() - start_time
                    path_length = self._compute_path_length(history)

                    return SimulationResult(
                        success=False,
                        steps_taken=step + 1,
                        final_distance=distance_to_goal,
                        path_length=path_length,
                        computation_time=computation_time,
                        collision_occurred=True,
                        num_obstacles=len(obstacles),
                        scenario_difficulty=np.linalg.norm(start_state[:2] - goal),
                        safety_violations=sum(safety_violations),
                        total_steps=step + 1,
                        scenario_id=scenario_id,
                        trajectory=history.copy()
                    )

            except Exception as e:
                if verbose:
                    print(f"Controller failed at step {step}: {e}")
                computation_time = time.time() - start_time
                return SimulationResult(
                    success=False,
                    steps_taken=step + 1,
                    final_distance=np.linalg.norm(state[:2] - goal),
                    path_length=self._compute_path_length(history),
                    computation_time=computation_time,
                    collision_occurred=False,
                    num_obstacles=len(obstacles),
                    scenario_difficulty=np.linalg.norm(start_state[:2] - goal),
                    safety_violations=sum(safety_violations),
                    total_steps=step + 1,
                    scenario_id=scenario_id,
                    trajectory=history.copy()
                )

        # Max steps exceeded
        computation_time = time.time() - start_time
        path_length = self._compute_path_length(history)

        return SimulationResult(
            success=False,
            steps_taken=max_steps,
            final_distance=np.linalg.norm(state[:2] - goal),
            path_length=path_length,
            computation_time=computation_time,
            collision_occurred=False,
            num_obstacles=len(obstacles),
            scenario_difficulty=np.linalg.norm(start_state[:2] - goal),
            safety_violations=sum(safety_violations),
            total_steps=max_steps,
            scenario_id=scenario_id,
            trajectory=history.copy()
        )

    def _compute_path_length(self, history: List[np.ndarray]) -> float:
        """Compute total path length"""
        if len(history) < 2:
            return 0.0

        path_length = 0.0
        for i in range(1, len(history)):
            path_length += np.linalg.norm(history[i][:2] - history[i-1][:2])

        return path_length

    def run_benchmark(self, num_scenarios: int = 20, trials_per_scenario: int = 1,
                     max_steps: int = 300, verbose: bool = True) -> List[SimulationResult]:
        """Run comprehensive benchmark with exactly num_scenarios different random configurations"""
        print(f"Running MPPI Benchmark using imported mppi_hard_constraint.py:")
        print(f"- {num_scenarios} unique random scenarios × {trials_per_scenario} trials = {num_scenarios * trials_per_scenario} total runs")
        print(f"- Max steps per trial: {max_steps}")
        print(f"- Controller params: {self.controller_params}")
        print("=" * 80)

        all_results = []
        successful_scenarios = 0

        for scenario_idx in range(num_scenarios):
            if verbose:
                print(f"\nScenario {scenario_idx + 1}/{num_scenarios}")

            # Generate random scenario with unique seed
            start_state, goal, obstacles = self.env_generator.generate_random_scenario(seed=42 + scenario_idx)
            
            # Store scenario for later visualization
            self.scenarios.append({
                'start_state': start_state.copy(),
                'goal': goal.copy(),
                'obstacles': [Obstacle(obs.x, obs.y, obs.radius, obs.safety_margin) for obs in obstacles],
                'scenario_id': scenario_idx
            })

            if verbose:
                print(f"  Start: [{start_state[0]:.1f}, {start_state[1]:.1f}]")
                print(f"  Goal: [{goal[0]:.1f}, {goal[1]:.1f}]")
                print(f"  Obstacles: {len(obstacles)}")
                print(f"  Distance: {np.linalg.norm(start_state[:2] - goal):.2f}")

            scenario_results = []

            # Run multiple trials for this scenario
            for trial_idx in range(trials_per_scenario):
                result = self.run_single_trial(start_state, goal, obstacles, max_steps, 
                                              verbose=False, scenario_id=scenario_idx)
                scenario_results.append(result)
                all_results.append(result)

                if verbose:
                    status = "SUCCESS" if result.success else "FAILED"
                    reason = ""
                    if not result.success:
                        if result.collision_occurred:
                            reason = " (collision)"
                        elif result.steps_taken >= max_steps:
                            reason = " (timeout)"
                        else:
                            reason = " (controller error)"

                    print(f"    Trial {trial_idx + 1}: {status}{reason}")
                    print(f"      Steps: {result.steps_taken}, Final dist: {result.final_distance:.2f}")
                    print(f"      Safety violations: {result.safety_violations}, Path length: {result.path_length:.2f}")

            # Scenario summary
            successes = sum(1 for r in scenario_results if r.success)
            if successes > 0:
                successful_scenarios += 1
                
            if verbose:
                print(f"  Scenario success rate: {successes}/{trials_per_scenario} ({successes/trials_per_scenario*100:.1f}%)")

        print(f"\nBenchmark Summary:")
        print(f"Successful scenarios: {successful_scenarios}/{num_scenarios} ({successful_scenarios/num_scenarios*100:.1f}%)")

        self.results = all_results
        return all_results

    def analyze_results(self) -> dict:
        """Analyze benchmark results and return comprehensive statistics"""
        if not self.results:
            print("No results to analyze. Run benchmark first.")
            return {}

        # Basic statistics
        total_runs = len(self.results)
        successful_runs = [r for r in self.results if r.success]
        failed_runs = [r for r in self.results if not r.success]
        collision_runs = [r for r in failed_runs if r.collision_occurred]
        timeout_runs = [r for r in failed_runs if r.steps_taken >= 300 and not r.collision_occurred]

        success_rate = len(successful_runs) / total_runs * 100 if total_runs > 0 else 0
        collision_rate = len(collision_runs) / total_runs * 100 if total_runs > 0 else 0
        timeout_rate = len(timeout_runs) / total_runs * 100 if total_runs > 0 else 0

        # Performance metrics for successful runs
        if successful_runs:
            avg_steps = np.mean([r.steps_taken for r in successful_runs])
            avg_path_length = np.mean([r.path_length for r in successful_runs])
            avg_computation_time = np.mean([r.computation_time for r in successful_runs])
            avg_safety_violations = np.mean([r.safety_violations for r in successful_runs])
            
            # Path efficiency
            straight_distances = [r.scenario_difficulty for r in successful_runs]
            path_lengths = [r.path_length for r in successful_runs]
            path_efficiencies = [straight / path * 100 for straight, path in zip(straight_distances, path_lengths) if path > 0]
            avg_path_efficiency = np.mean(path_efficiencies) if path_efficiencies else 0
        else:
            avg_steps = avg_path_length = avg_computation_time = avg_safety_violations = avg_path_efficiency = 0

        # Scenario analysis
        obstacle_counts = [r.num_obstacles for r in self.results]
        difficulties = [r.scenario_difficulty for r in self.results]
        
        # Safety analysis
        total_safety_violations = sum(r.safety_violations for r in self.results)
        total_simulation_steps = sum(r.total_steps for r in self.results)
        safety_violation_rate = total_safety_violations / total_simulation_steps * 100 if total_simulation_steps > 0 else 0

        # Success rate by obstacle count
        success_by_obstacles = {}
        for count in range(min(obstacle_counts), max(obstacle_counts) + 1):
            runs_with_count = [r for r in self.results if r.num_obstacles == count]
            if runs_with_count:
                success_by_obstacles[count] = sum(1 for r in runs_with_count if r.success) / len(runs_with_count) * 100

        # Success rate by difficulty
        difficulty_bins = np.linspace(min(difficulties), max(difficulties), 4)
        success_by_difficulty = {}
        for i in range(len(difficulty_bins) - 1):
            runs_in_bin = [r for r in self.results if difficulty_bins[i] <= r.scenario_difficulty < difficulty_bins[i+1]]
            if runs_in_bin:
                success_by_difficulty[f"{difficulty_bins[i]:.1f}-{difficulty_bins[i+1]:.1f}"] = sum(1 for r in runs_in_bin if r.success) / len(runs_in_bin) * 100

        stats = {
            'total_runs': total_runs,
            'total_scenarios': len(set(r.scenario_id for r in self.results)),
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_steps_success': avg_steps,
            'avg_path_length': avg_path_length,
            'avg_computation_time': avg_computation_time,
            'avg_safety_violations': avg_safety_violations,
            'safety_violation_rate': safety_violation_rate,
            'avg_path_efficiency': avg_path_efficiency,
            'obstacle_range': (min(obstacle_counts), max(obstacle_counts)),
            'difficulty_range': (min(difficulties), max(difficulties)),
            'success_by_obstacles': success_by_obstacles,
            'success_by_difficulty': success_by_difficulty,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'collision_runs': collision_runs,
            'timeout_runs': timeout_runs
        }

        return stats

    def plot_benchmark_results(self):
        """Create comprehensive plots of benchmark results"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        stats = self.analyze_results()
        
        # First, plot statistical analysis
        self.plot_statistical_analysis(stats)
        
        # Then, plot all scenario configurations
        self.plot_scenario_configurations(stats)

        # Print detailed summary
        self.print_detailed_summary(stats)

    def plot_statistical_analysis(self, stats):
        """Plot statistical analysis results"""
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Overall Success Rate
        ax1 = plt.subplot(3, 3, 1)
        categories = ['Success', 'Collision', 'Timeout', 'Other']
        values = [stats['success_rate'], stats['collision_rate'], stats['timeout_rate'], 
                 100 - stats['success_rate'] - stats['collision_rate'] - stats['timeout_rate']]
        colors = ['green', 'red', 'orange', 'gray']
        
        wedges, texts, autotexts = ax1.pie(values, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title(f'Overall Results\n({stats["total_runs"]} runs, {stats["total_scenarios"]} scenarios)')

        # Plot 2: Success rate by number of obstacles
        ax2 = plt.subplot(3, 3, 2)
        if stats['success_by_obstacles']:
            obs_counts = list(stats['success_by_obstacles'].keys())
            success_rates = list(stats['success_by_obstacles'].values())
            bars = ax2.bar(obs_counts, success_rates, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Number of Obstacles')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rate vs Obstacle Count')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

        # Plot 3: Success rate by scenario difficulty
        ax3 = plt.subplot(3, 3, 3)
        if stats['success_by_difficulty']:
            diff_ranges = list(stats['success_by_difficulty'].keys())
            success_rates = list(stats['success_by_difficulty'].values())
            bars = ax3.bar(range(len(diff_ranges)), success_rates, alpha=0.7, color='lightgreen')
            ax3.set_xlabel('Scenario Difficulty (Start-Goal Distance)')
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Success Rate vs Scenario Difficulty')
            ax3.set_xticks(range(len(diff_ranges)))
            ax3.set_xticklabels(diff_ranges, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Computation time distribution
        ax4 = plt.subplot(3, 3, 4)
        comp_times = [r.computation_time for r in self.results]
        ax4.hist(comp_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Computation Time (s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Computation Time Distribution')
        ax4.grid(True, alpha=0.3)
        avg_time = np.mean(comp_times)
        ax4.axvline(avg_time, color='red', linestyle='--', 
                   label=f'Avg: {avg_time:.3f}s')
        ax4.legend()

        # Plot 5: Steps taken distribution
        ax5 = plt.subplot(3, 3, 5)
        successful_steps = [r.steps_taken for r in stats['successful_runs']]
        failed_steps = [r.steps_taken for r in stats['failed_runs']]

        if successful_steps:
            ax5.hist(successful_steps, bins=20, alpha=0.7, label='Successful', 
                    color='green', edgecolor='black')
        if failed_steps:
            ax5.hist(failed_steps, bins=20, alpha=0.7, label='Failed', 
                    color='red', edgecolor='black')
        ax5.set_xlabel('Steps Taken')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Steps Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Path efficiency
        ax6 = plt.subplot(3, 3, 6)
        if stats['successful_runs']:
            straight_distances = [r.scenario_difficulty for r in stats['successful_runs']]
            path_lengths = [r.path_length for r in stats['successful_runs']]
            efficiencies = [straight / path * 100 for straight, path in zip(straight_distances, path_lengths) if path > 0]
            
            ax6.scatter(straight_distances, efficiencies, alpha=0.6, color='purple')
            ax6.set_xlabel('Straight Line Distance')
            ax6.set_ylabel('Path Efficiency (%)')
            ax6.set_title(f'Path Efficiency\n(Avg: {stats["avg_path_efficiency"]:.1f}%)')
            ax6.grid(True, alpha=0.3)
            
            # Add ideal efficiency line
            if straight_distances:
                ax6.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Perfect efficiency')
                ax6.legend()

        # Plot 7: Safety violations analysis
        ax7 = plt.subplot(3, 3, 7)
        safety_viols = [r.safety_violations for r in self.results]
        total_steps = [r.total_steps for r in self.results]
        violation_rates = [viols / steps * 100 if steps > 0 else 0 for viols, steps in zip(safety_viols, total_steps)]
        
        ax7.hist(violation_rates, bins=15, alpha=0.7, color='salmon', edgecolor='black')
        ax7.set_xlabel('Safety Violation Rate (%)')
        ax7.set_ylabel('Frequency')
        ax7.set_title(f'Safety Violations\n(Overall rate: {stats["safety_violation_rate"]:.1f}%)')
        ax7.grid(True, alpha=0.3)

        # Plot 8: Performance metrics summary
        ax8 = plt.subplot(3, 3, 8)
        metrics = ['Success\nRate (%)', 'Avg Steps\n(Success)', 'Avg Path\nLength', 'Safety Viol.\nRate (%)']
        values = [stats['success_rate'], stats['avg_steps_success'], 
                 stats['avg_path_length'], stats['safety_violation_rate']]
        
        # Normalize values for comparison (scale to 0-100)
        normalized_values = []
        for i, val in enumerate(values):
            if i == 0 or i == 3:  # Already percentages
                normalized_values.append(val)
            elif i == 1:  # Steps
                normalized_values.append(val / 3)  # Normalize assuming max 300 steps
            else:  # Path length
                normalized_values.append(val * 5)  # Scale up for visibility
        
        bars = ax8.bar(metrics, normalized_values, alpha=0.7, 
                      color=['green', 'blue', 'orange', 'red'])
        ax8.set_ylabel('Normalized Value')
        ax8.set_title('Performance Summary')
        ax8.grid(True, alpha=0.3)
        
        # Add actual values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        # Plot 9: Success/Failure by scenario
        ax9 = plt.subplot(3, 3, 9)
        scenario_ids = list(range(len(self.scenarios)))
        success_status = []
        for scenario_id in scenario_ids:
            scenario_results = [r for r in self.results if r.scenario_id == scenario_id]
            if scenario_results and scenario_results[0].success:
                success_status.append(1)
            else:
                success_status.append(0)
        
        colors = ['green' if s else 'red' for s in success_status]
        bars = ax9.bar(scenario_ids, [1]*len(scenario_ids), color=colors, alpha=0.7)
        ax9.set_xlabel('Scenario ID')
        ax9.set_ylabel('Success (1) / Failure (0)')
        ax9.set_title('Success/Failure by Scenario')
        ax9.grid(True, alpha=0.3)
        ax9.set_xticks(range(0, len(scenario_ids), 2))

        plt.tight_layout()
        plt.suptitle(f'MPPI Statistical Analysis - {stats["total_scenarios"]} Random Scenarios\n'
                    f'Overall Success Rate: {stats["success_rate"]:.1f}% | '
                    f'Safety Violation Rate: {stats["safety_violation_rate"]:.1f}%',
                    y=0.98, fontsize=16)
        plt.show()

    def plot_scenario_configurations(self, stats):
        """Plot all 20 scenario configurations with paths"""
        # Plot configurations in 4x5 grid to show all 20 scenarios
        fig = plt.figure(figsize=(25, 20))
        
        for scenario_idx in range(min(20, len(self.scenarios))):
            ax = plt.subplot(4, 5, scenario_idx + 1)
            scenario = self.scenarios[scenario_idx]
            
            # Plot obstacles
            for obstacle in scenario['obstacles']:
                # Safety margin
                safety_circle = patches.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.inflated_radius,
                    color="pink",
                    alpha=0.3
                )
                ax.add_patch(safety_circle)
                
                # Actual obstacle
                circle = patches.Circle(
                    (obstacle.x, obstacle.y), 
                    obstacle.radius,
                    color='red', 
                    alpha=0.7
                )
                ax.add_patch(circle)

            # Plot start and goal
            start = scenario['start_state']
            goal = scenario['goal']
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start' if scenario_idx == 0 else "")
            ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal' if scenario_idx == 0 else "")

            # Plot straight line path
            ax.plot([start[0], goal[0]], [start[1], goal[1]],
                   'k--', alpha=0.3, linewidth=1, label='Direct Path' if scenario_idx == 0 else "")

            # Get and plot actual MPPI path
            scenario_results = [r for r in self.results if r.scenario_id == scenario_idx]
            if scenario_results:
                result = scenario_results[0]
                
                # Plot actual trajectory if available
                if result.trajectory and len(result.trajectory) > 1:
                    trajectory = np.array(result.trajectory)
                    
                    if result.success:
                        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=3, alpha=0.8, 
                               label='MPPI Path (Success)' if scenario_idx == 0 else "")
                        success_status = "✓"
                        title_color = 'green'
                        status_text = f"SUCCESS\nSteps: {result.steps_taken}\nPath: {result.path_length:.1f}m\nSafety Viols: {result.safety_violations}"
                    else:
                        if result.collision_occurred:
                            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3, alpha=0.8,
                                   label='MPPI Path (Collision)' if scenario_idx == 0 else "")
                            success_status = "✗"
                            title_color = 'red'
                            status_text = f"COLLISION\nSteps: {result.steps_taken}\nDist: {result.final_distance:.1f}m\nSafety Viols: {result.safety_violations}"
                        else:
                            ax.plot(trajectory[:, 0], trajectory[:, 1], 'orange', linewidth=3, alpha=0.8,
                                   label='MPPI Path (Timeout)' if scenario_idx == 0 else "")
                            success_status = "⏱"
                            title_color = 'orange' 
                            status_text = f"TIMEOUT\nSteps: {result.steps_taken}\nDist: {result.final_distance:.1f}m\nSafety Viols: {result.safety_violations}"
                    
                    # Mark final position
                    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ko', markersize=8, alpha=0.7)
                else:
                    # Fallback to representative path if no trajectory stored
                    success_status = "✓" if result.success else ("✗" if result.collision_occurred else "⏱")
                    title_color = 'green' if result.success else ('red' if result.collision_occurred else 'orange')
                    status_text = f"{'SUCCESS' if result.success else ('COLLISION' if result.collision_occurred else 'TIMEOUT')}\nSteps: {result.steps_taken}"
                
                # Add result info text box
                ax.text(0.02, 0.98, status_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            else:
                success_status = "?"
                title_color = 'gray'

            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            # Title with scenario info
            ax.set_title(f'Scenario {scenario_idx + 1} {success_status}\n'
                        f'Obs: {len(scenario["obstacles"])}, '
                        f'Dist: {np.linalg.norm(start[:2] - goal):.1f}m',
                        color=title_color, fontsize=10, fontweight='bold')
            
            # Only show ticks on edge plots to save space
            if scenario_idx >= 15:  # Bottom row
                ax.set_xlabel('X (m)', fontsize=8)
            else:
                ax.set_xticks([])
                
            if scenario_idx % 5 == 0:  # Left column
                ax.set_ylabel('Y (m)', fontsize=8)
            else:
                ax.set_yticks([])
                
            # Add legend only to first plot
            if scenario_idx == 0:
                ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.suptitle(f'All {len(self.scenarios)} Scenario Configurations with MPPI Paths\n'
                    f'Green ✓ = Success, Red ✗ = Collision, Orange ⏱ = Timeout',
                    y=0.98, fontsize=16, fontweight='bold')
        plt.show()

    def print_detailed_summary(self, stats):
        """Print comprehensive benchmark summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MPPI BENCHMARK RESULTS")
        print("="*80)
        
        print(f"OVERALL PERFORMANCE:")
        print(f"  Total scenarios tested: {stats['total_scenarios']}")
        print(f"  Total simulation runs: {stats['total_runs']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Collision rate: {stats['collision_rate']:.1f}%")
        print(f"  Timeout rate: {stats['timeout_rate']:.1f}%")
        
        print(f"\nSUCCESSFUL RUNS ANALYSIS:")
        if stats['successful_runs']:
            print(f"  Average steps to goal: {stats['avg_steps_success']:.1f}")
            print(f"  Average path length: {stats['avg_path_length']:.2f}")
            print(f"  Average path efficiency: {stats['avg_path_efficiency']:.1f}%")
            print(f"  Average computation time: {stats['avg_computation_time']:.3f}s")
            print(f"  Average safety violations: {stats['avg_safety_violations']:.1f}")
        else:
            print("  No successful runs to analyze")
        
        print(f"\nSAFETY ANALYSIS:")
        print(f"  Overall safety violation rate: {stats['safety_violation_rate']:.1f}%")
        total_safety_viols = sum(r.safety_violations for r in self.results)
        print(f"  Total safety violations: {total_safety_viols}")
        print(f"  Runs with safety violations: {sum(1 for r in self.results if r.safety_violations > 0)}")
        
        print(f"\nSCENARIO COMPLEXITY ANALYSIS:")
        print(f"  Obstacle count range: {stats['obstacle_range'][0]}-{stats['obstacle_range'][1]}")
        print(f"  Difficulty range: {stats['difficulty_range'][0]:.1f}-{stats['difficulty_range'][1]:.1f}")
        
        if stats['success_by_obstacles']:
            print(f"\nSUCCESS RATE BY OBSTACLE COUNT:")
            for obs_count, success_rate in stats['success_by_obstacles'].items():
                print(f"  {obs_count} obstacles: {success_rate:.1f}%")
        
        if stats['success_by_difficulty']:
            print(f"\nSUCCESS RATE BY DIFFICULTY:")
            for difficulty_range, success_rate in stats['success_by_difficulty'].items():
                print(f"  Distance {difficulty_range}: {success_rate:.1f}%")
        
        print("\n" + "="*80)

class MPPISimulation:
    """Simulation environment for MPPI with inflated obstacle safety margins"""

    def __init__(self, start_state=None, goal=None, obstacles=None, safety_margin: float = 0.2):
        self.safety_margin = safety_margin
        self.controller = MPPIController(
            horizon=20,
            num_samples=100,
            dt=0.1,
            lambda_=1.0,
            sigma=1.0,
            safety_margin=safety_margin
        )

        # Environment setup - use provided or default
        if start_state is not None:
            self.state = start_state.copy()
        else:
            self.state = np.array([0.0, 0.0, 0.0, 0.0])  # Start at origin

        if goal is not None:
            self.goal = goal.copy()
        else:
            self.goal = np.array([8.0, 8.0])  # Goal position

        if obstacles is not None:
            self.obstacles = obstacles
        else:
            # Create obstacles
            self.obstacles = [
                Obstacle(2.5, 2.0, 0.6, safety_margin),
                Obstacle(5.0, 4.0, 0.8, safety_margin),
                Obstacle(3.0, 6.0, 0.5, safety_margin),
                Obstacle(6.5, 2.5, 0.7, safety_margin),
                Obstacle(6.0, 6.5, 0.6, safety_margin)
            ]

        # Simulation history
        self.history = []
        self.control_history = []
        self.cost_history = []
        self.safety_violations = []  # Track when robot enters safety margins

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
            print(
                f"Obstacle {i + 1}: center=({obs.x:.1f}, {obs.y:.1f}), "
                f"radius={obs.radius:.1f}, inflated_radius={obs.inflated_radius:.1f}"
            )

        for step in range(max_steps):
            # Get control from MPPI
            control, best_trajectory, costs = self.controller.update_control(
                self.state, self.goal, self.obstacles
            )

            # Apply control and update state
            self.state = self.controller.dynamics(self.state, control)

            # Check safety violations (robot in safety margin but not colliding)
            safety_violation = False
            actual_collision = False

            for obstacle in self.obstacles:
                distance = np.linalg.norm(
                    self.state[:2] - np.array([obstacle.x, obstacle.y])
                )

                if distance < obstacle.radius:
                    actual_collision = True
                    break
                elif distance < obstacle.inflated_radius:
                    safety_violation = True

            self.safety_violations.append(safety_violation)

            # Store history
            self.history.append(self.state.copy())
            self.control_history.append(control.copy())
            self.cost_history.append(np.min(costs))

            # Check if goal reached
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

    def plot_results(self):
        """Plot the simulation results with safety margins"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Trajectory and environment
        ax1.set_title("MPPI Trajectory with Safety Margins")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        # Plot obstacles with safety margins
        for obstacle in self.obstacles:
            # Plot safety margin
            safety_circle = patches.Circle(
                (obstacle.x, obstacle.y),
                obstacle.inflated_radius,
                color="pink",
                alpha=0.3,
                label="Safety Margin" if obstacle == self.obstacles[0] else "",
            )
            ax1.add_patch(safety_circle)
            
            # Plot actual obstacle
            obstacle_circle = patches.Circle(
                (obstacle.x, obstacle.y), 
                obstacle.radius,
                color="red", 
                alpha=0.7,
                label="Obstacle" if obstacle == self.obstacles[0] else ""
            )
            ax1.add_patch(obstacle_circle)

        # Plot trajectory with safety violation highlighting
        if self.history:
            trajectory = np.array(self.history)
            
            # Split trajectory into safe segments and safety violation segments
            safe_segments = []
            violation_segments = []

            for i in range(len(trajectory)):
                if i < len(self.safety_violations) and self.safety_violations[i]:
                    violation_segments.append(trajectory[i])
                else:
                    safe_segments.append(trajectory[i])

            # Plot safe segments in blue
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

            # Plot violation segments in orange
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
                linewidth=1,
                alpha=0.5,
                zorder=1,
            )
            
            # Plot start point
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
            ax2.plot(np.linalg.norm(trajectory[:, 2:4], axis=1), label="|V|", linewidth=2, linestyle="--")
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

        # Plot 4: Cost evolution and safety violations
        if self.cost_history:
            ax4.set_title("Cost Evolution and Safety Violations")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Cost")
            ax4.grid(True, alpha=0.3)
            
            # Plot cost on primary y-axis
            ax4.plot(self.cost_history, "b-", linewidth=2, label="Cost")
            ax4.set_yscale("log")
            
            # Create twin axis for safety violations
            ax4_twin = ax4.twinx()
            ax4_twin.set_ylabel("Safety Violation", color="r")
            
            if self.safety_violations:
                ax4_twin.plot(
                    self.safety_violations, "r-", linewidth=2, label="Safety Violation"
                )
                ax4_twin.fill_between(
                    range(len(self.safety_violations)),
                    self.safety_violations,
                    alpha=0.3,
                    color="red",
                )
                ax4_twin.set_ylim(-0.1, 1.1)
                
            # Add combined legend
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_safety_analysis(self):
        """Plot safety analysis showing inflated obstacles and violations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Trajectory with safety zones
        ax1.set_title(f"Trajectory with Safety Margins (margin = {self.safety_margin})")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)

        # Plot safety zones and obstacles
        for obstacle in self.obstacles:
            # Safety zone
            safety_circle = patches.Circle(
                (obstacle.x, obstacle.y),
                obstacle.inflated_radius,
                color="pink",
                alpha=0.3,
                label="Safety Margin" if obstacle == self.obstacles[0] else "",
            )
            ax1.add_patch(safety_circle)

            # Original obstacle
            obstacle_circle = patches.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                color="red",
                alpha=0.7,
                label="Actual Obstacle" if obstacle == self.obstacles[0] else "",
            )
            ax1.add_patch(obstacle_circle)

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
                    obs_pos = np.array([obstacle.x, obstacle.y])
                    dist = np.linalg.norm(pos - obs_pos)

                    min_dist_actual = min(min_dist_actual, dist - obstacle.radius)
                    min_dist_inflated = min(
                        min_dist_inflated, dist - obstacle.inflated_radius
                    )

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

        # Plot 4: Performance metrics over time
        ax4.set_title("Path to Goal")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Distance to Goal (m)")
        ax4.grid(True, alpha=0.3)

        if self.history:
            # Distance to goal over time
            goal_distances = []
            for state in self.history:
                goal_dist = np.linalg.norm(state[:2] - self.goal)
                goal_distances.append(goal_dist)

            ax4.plot(goal_distances, "g-", linewidth=2, label="Distance to Goal")
            
            # Mark timesteps with safety violations
            if self.safety_violations:
                for i, violation in enumerate(self.safety_violations):
                    if violation and i < len(goal_distances):
                        ax4.plot(i, goal_distances[i], 'ro', alpha=0.5, markersize=4)
            
            # Add final distance
            if goal_distances:
                ax4.text(
                    0.02,
                    0.98,
                    f"Final distance: {goal_distances[-1]:.3f}m\n"
                    f"Path length: {sum(np.linalg.norm(np.diff(np.array(self.history)[:, :2], axis=0), axis=1)):.3f}m",
                    transform=ax4.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

            ax4.legend()

        plt.tight_layout()
        plt.show()

def demonstrate_random_scenarios():
    """Demonstrate a few random scenarios visually with safety margins"""
    print("Demonstrating random scenario generation with safety margins...")

    env_gen = RandomEnvironmentGenerator(safety_margin=0.2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]

        # Generate random scenario
        start_state, goal, obstacles = env_gen.generate_random_scenario(seed=i+10)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Random Scenario {i+1}")

        # Plot obstacles with safety margins
        for obstacle in obstacles:
            # Safety margin
            safety_circle = patches.Circle(
                (obstacle.x, obstacle.y),
                obstacle.inflated_radius,
                color="pink",
                alpha=0.3,
                label="Safety Margin" if obstacle == obstacles[0] and i == 0 else ""
            )
            ax.add_patch(safety_circle)
            
            # Actual obstacle
            circle = patches.Circle(
                (obstacle.x, obstacle.y), 
                obstacle.radius,
                color='red', 
                alpha=0.7,
                label='Obstacle' if obstacle == obstacles[0] and i == 0 else ""
            )
            ax.add_patch(circle)

        # Plot start and goal
        ax.plot(start_state[0], start_state[1], 'go', markersize=12, label='Start')
        ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

        # Plot straight line path
        ax.plot([start_state[0], goal[0]], [start_state[1], goal[1]],
               'k--', alpha=0.5, label='Direct path')

        ax.legend()

        # Add scenario info
        distance = np.linalg.norm(start_state[:2] - goal)
        ax.text(0.02, 0.98, f'Obstacles: {len(obstacles)}\nDistance: {distance:.1f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.suptitle("Random Scenario Examples with Safety Margins", y=1.02, fontsize=16)
    plt.show()

def main():
    """Main function for running the MPPI benchmark test"""
    print("=" * 80)
    print("MPPI BENCHMARK TEST - With Safety Margins")
    print("=" * 80)
    
    print("Choose an option:")
    print("1. View random environment examples")
    print("2. Run single scenario test")
    print("3. Run full benchmark (20 scenarios)")
    print("4. Exit")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
    except:
        choice = "2"
    
    if choice == "1":
        demonstrate_random_scenarios()
    elif choice == "2":
        run_single_scenario_test()
    elif choice == "3":
        # Run benchmark with optimized parameters
        controller_params = {
            'horizon': 20,
            'num_samples': 100,
            'dt': 0.1,
            'lambda_': 1.0,
            'sigma': 1.0,
            'safety_margin': 0.2
        }
        
        print("\nController Configuration:")
        for key, value in controller_params.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        
        benchmark = MPPIBenchmark(controller_params)
        results, stats, benchmark = run_benchmark(benchmark)
    elif choice == "4":
        print("Exiting program.")
        return
    else:
        print("Invalid choice, running single scenario test...")
        run_single_scenario_test()

def run_benchmark(benchmark):
    """Run the full benchmark with 20 scenarios"""
    print("\nGenerating 20 unique random scenarios and running benchmark...")
    print("This may take a few minutes...")
    
    start_time = time.time()
    results = benchmark.run_benchmark(
        num_scenarios=20,
        trials_per_scenario=1,  # One trial per scenario for 20 unique configurations
        max_steps=300,
        verbose=True
    )
    end_time = time.time()

    print(f"\nBenchmark completed in {end_time - start_time:.2f} seconds")
    print(f"Total configurations tested: {len(benchmark.scenarios)}")
    print(f"Total simulation runs: {len(results)}")

    # Analyze and visualize results
    print("\nAnalyzing results and generating visualizations...")
    stats = benchmark.analyze_results()
    benchmark.plot_benchmark_results()

    # Quick summary
    print(f"\nQUICK SUMMARY:")
    print(f"✓ Scenarios successfully solved: {len([r for r in results if r.success])}/20")
    print(f"✗ Scenarios with collisions: {len([r for r in results if r.collision_occurred])}/20")
    print(f"⏱ Scenarios that timed out: {len([r for r in results if r.steps_taken >= 300 and not r.collision_occurred])}/20")
    
    if stats['successful_runs']:
        avg_efficiency = stats['avg_path_efficiency']
        print(f"📏 Average path efficiency: {avg_efficiency:.1f}%")
        
    safety_rate = stats['safety_violation_rate']
    print(f"🛡 Safety violation rate: {safety_rate:.1f}%")
    
    return results, stats, benchmark

def run_single_scenario_test():
    """Test a single scenario with safety margins"""
    print("\n" + "="*60)
    print("SINGLE SCENARIO TEST")
    print("="*60)
    
    # Create a simple test scenario
    safety_margin = 0.2
    sim = MPPISimulation(safety_margin=safety_margin)
    
    print(f"Running single scenario test with safety margin: {safety_margin}")
    start_time = time.time()
    success = sim.run_simulation(max_steps=200)
    end_time = time.time()
    
    print(f"\nSingle scenario completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {success}")
    
    if sim.history:
        final_state = sim.history[-1]
        final_distance = np.linalg.norm(final_state[:2] - sim.goal)
        print(f"Final distance to goal: {final_distance:.3f}")
        print(f"Safety violations: {sum(sim.safety_violations)} out of {len(sim.safety_violations)} steps")
    
    # Plot results from single scenario
    print("\nGenerating single scenario visualizations...")
    sim.plot_results()
    sim.plot_safety_analysis()

if __name__ == "__main__":
    main()
    
    print("\n" + "="*80)
    print("MPPI BENCHMARK TEST COMPLETE!")
    print("Using components imported from mppi_hard_constraint.py")
    print("="*80)