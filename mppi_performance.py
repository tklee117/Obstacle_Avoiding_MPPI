import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import random

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
                 min_separation: float = 0.5,
                 safety_margin: float = 0.2):
        """
        Initialize environment generator

        Args:
            world_size: (width, height) of the world
            min_obstacles: Minimum number of obstacles
            max_obstacles: Maximum number of obstacles
            min_radius: Minimum obstacle radius
            max_radius: Maximum obstacle radius
            min_separation: Minimum separation between obstacles and start/goal
            safety_margin: Safety margin for obstacles
        """
        self.world_size = world_size
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_separation = min_separation
        self.safety_margin = safety_margin

    def generate_random_scenario(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Obstacle]]:
        """
        Generate a random scenario with start, goal, and obstacles

        Args:
            seed: Random seed for reproducibility

        Returns:
            start_state: Initial robot state [x, y, vx, vy]
            goal: Goal position [x, y]
            obstacles: List of obstacles
        """
        # if seed is not None:
        #     np.random.seed(seed)
        #     random.seed(seed)

        max_attempts = 100

        for attempt in range(max_attempts):
            # Generate random start and goal positions
            start_pos = np.array([
                np.random.uniform(0.5, self.world_size[0] - 0.5),
                np.random.uniform(0.5, self.world_size[1] - 0.5)
            ])

            goal_pos = np.array([
                np.random.uniform(0.5, self.world_size[0] - 0.5),
                np.random.uniform(0.5, self.world_size[1] - 0.5)
            ])

            # Ensure start and goal are far enough apart
            if np.linalg.norm(goal_pos - start_pos) < 3.0:
                continue

            # Generate random number of obstacles
            num_obstacles = np.random.randint(self.min_obstacles, self.max_obstacles + 1)
            obstacles = []

            # Generate obstacles
            obstacles_generated = 0
            obstacle_attempts = 0

            while obstacles_generated < num_obstacles and obstacle_attempts < 50:
                obstacle_attempts += 1

                # Random position and radius
                pos = np.array([
                    np.random.uniform(self.min_radius, self.world_size[0] - self.min_radius),
                    np.random.uniform(self.min_radius, self.world_size[1] - self.min_radius)
                ])
                radius = np.random.uniform(self.min_radius, self.max_radius)

                # Check if obstacle is too close to start or goal
                if (np.linalg.norm(pos - start_pos) < radius + self.min_separation or
                    np.linalg.norm(pos - goal_pos) < radius + self.min_separation):
                    continue

                # Check if obstacle overlaps with existing obstacles
                valid = True
                for existing_obstacle in obstacles:
                    existing_pos = np.array([existing_obstacle.x, existing_obstacle.y])
                    min_distance = radius + existing_obstacle.radius + 0.2  # Small buffer
                    if np.linalg.norm(pos - existing_pos) < min_distance:
                        valid = False
                        break

                if valid:
                    obstacles.append(Obstacle(pos[0], pos[1], radius, self.safety_margin))
                    obstacles_generated += 1

            # Check if there's a feasible path (simple line-of-sight check)
            if self._check_line_of_sight(start_pos, goal_pos, obstacles):
                start_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
                return start_state, goal_pos, obstacles

        # Fallback: return a simple scenario if generation fails
        print("Warning: Could not generate complex scenario, using simple fallback")
        start_state = np.array([0.5, 0.5, 0.0, 0.0])
        goal_pos = np.array([self.world_size[0] - 0.5, self.world_size[1] - 0.5])
        obstacles = [Obstacle(self.world_size[0]/2, self.world_size[1]/2, 0.5, self.safety_margin)]
        return start_state, goal_pos, obstacles

    def _check_line_of_sight(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Obstacle]) -> bool:
        """Check if there's a clear line of sight between start and goal"""
        direction = goal - start
        distance = np.linalg.norm(direction)
        if distance == 0:
            return True

        direction_norm = direction / distance

        # Sample points along the line
        num_samples = int(distance * 10)  # 10 samples per unit distance
        for i in range(num_samples):
            t = i / num_samples
            point = start + t * direction

            # Check if point is inside any obstacle
            for obstacle in obstacles:
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                if np.linalg.norm(point - obstacle_pos) < obstacle.radius:
                    return False

        return True

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
    scenario_difficulty: float  # Distance from start to goal

class MPPIBenchmark:
    """Comprehensive benchmarking suite for MPPI"""

    def __init__(self, controller_params: dict = None):
        """
        Initialize benchmark suite

        Args:
            controller_params: Parameters for MPPI controller
        """
        self.controller_params = controller_params or {
            'horizon': 20,
            'num_samples': 100,
            'dt': 0.1,
            'lambda_': 1.0,
            'sigma': 1.0,
            'safety_margin': 0.2
        }

        self.env_generator = RandomEnvironmentGenerator()
        self.results = []

    def run_single_trial(self, start_state: np.ndarray, goal: np.ndarray,
                        obstacles: List[Obstacle], max_steps: int = 200,
                        verbose: bool = False) -> SimulationResult:
        """Run a single MPPI trial"""
        # Create controller with safety margin
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
            # Get control from MPPI
            try:
                control, _, _ = controller.update_control(state, goal, obstacles)

                # Apply control and update state
                state = controller.dynamics(state, control)
                history.append(state.copy())

                # Check safety violations
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
                if distance_to_goal < 0.3:
                    # Success!
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
                        scenario_difficulty=np.linalg.norm(start_state[:2] - goal)
                    )

                # Check collision with obstacles (actual collision, not safety margin)
                if actual_collision:
                    # Collision!
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
                        scenario_difficulty=np.linalg.norm(start_state[:2] - goal)
                    )

            except Exception as e:
                if verbose:
                    print(f"Controller failed at step {step}: {e}")
                # Controller failure
                computation_time = time.time() - start_time
                return SimulationResult(
                    success=False,
                    steps_taken=step + 1,
                    final_distance=np.linalg.norm(state[:2] - goal),
                    path_length=self._compute_path_length(history),
                    computation_time=computation_time,
                    collision_occurred=False,
                    num_obstacles=len(obstacles),
                    scenario_difficulty=np.linalg.norm(start_state[:2] - goal)
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
            scenario_difficulty=np.linalg.norm(start_state[:2] - goal)
        )

    def _compute_path_length(self, history: List[np.ndarray]) -> float:
        """Compute total path length"""
        if len(history) < 2:
            return 0.0

        path_length = 0.0
        for i in range(1, len(history)):
            path_length += np.linalg.norm(history[i][:2] - history[i-1][:2])

        return path_length

    def run_benchmark(self, num_scenarios: int = 20, trials_per_scenario: int = 3,
                     max_steps: int = 200, verbose: bool = True) -> List[SimulationResult]:
        """
        Run comprehensive benchmark

        Args:
            num_scenarios: Number of different random scenarios
            trials_per_scenario: Number of trials per scenario
            max_steps: Maximum steps per trial
            verbose: Print progress

        Returns:
            List of all simulation results
        """
        print(f"Running MPPI Benchmark:")
        print(f"- {num_scenarios} scenarios × {trials_per_scenario} trials = {num_scenarios * trials_per_scenario} total runs")
        print(f"- Max steps per trial: {max_steps}")
        print(f"- Controller params: {self.controller_params}")
        print("=" * 60)

        all_results = []

        for scenario_idx in range(num_scenarios):
            if verbose:
                print(f"\nScenario {scenario_idx + 1}/{num_scenarios}")

            # Generate random scenario
            start_state, goal, obstacles = self.env_generator.generate_random_scenario(seed=scenario_idx)

            if verbose:
                print(f"  Start: {start_state[:2]}")
                print(f"  Goal: {goal}")
                print(f"  Obstacles: {len(obstacles)}")
                print(f"  Distance: {np.linalg.norm(start_state[:2] - goal):.2f}")

            scenario_results = []

            # Run multiple trials for this scenario
            for trial_idx in range(trials_per_scenario):
                result = self.run_single_trial(start_state, goal, obstacles, max_steps, verbose=False)
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

                    print(f"    Trial {trial_idx + 1}: {status}{reason} - {result.steps_taken} steps, {result.final_distance:.2f} dist")

            # Scenario summary
            successes = sum(1 for r in scenario_results if r.success)
            if verbose:
                print(f"  Scenario success rate: {successes}/{trials_per_scenario} ({successes/trials_per_scenario*100:.1f}%)")

        self.results = all_results
        return all_results

    def analyze_results(self) -> dict:
        """Analyze benchmark results and return statistics"""
        if not self.results:
            print("No results to analyze. Run benchmark first.")
            return {}

        # Basic statistics
        total_runs = len(self.results)
        successful_runs = [r for r in self.results if r.success]
        failed_runs = [r for r in self.results if not r.success]
        collision_runs = [r for r in failed_runs if r.collision_occurred]

        success_rate = len(successful_runs) / total_runs * 100
        collision_rate = len(collision_runs) / total_runs * 100

        # Performance metrics for successful runs
        if successful_runs:
            avg_steps = np.mean([r.steps_taken for r in successful_runs])
            avg_path_length = np.mean([r.path_length for r in successful_runs])
            avg_computation_time = np.mean([r.computation_time for r in successful_runs])
        else:
            avg_steps = avg_path_length = avg_computation_time = 0

        # Difficulty analysis
        obstacle_counts = [r.num_obstacles for r in self.results]
        difficulties = [r.scenario_difficulty for r in self.results]

        stats = {
            'total_runs': total_runs,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'avg_steps_success': avg_steps,
            'avg_path_length': avg_path_length,
            'avg_computation_time': avg_computation_time,
            'obstacle_range': (min(obstacle_counts), max(obstacle_counts)),
            'difficulty_range': (min(difficulties), max(difficulties)),
            'successful_runs': successful_runs,
            'failed_runs': failed_runs
        }

        return stats

    def plot_benchmark_results(self):
        """Create comprehensive plots of benchmark results"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        stats = self.analyze_results()

        fig = plt.figure(figsize=(20, 15))

        # Plot 1: Success rate by number of obstacles
        ax1 = plt.subplot(3, 4, 1)
        obstacle_counts = np.array([r.num_obstacles for r in self.results])
        successes_by_obstacles = {}

        for count in np.unique(obstacle_counts):
            runs_with_count = [r for r in self.results if r.num_obstacles == count]
            success_rate = sum(1 for r in runs_with_count if r.success) / len(runs_with_count) * 100
            successes_by_obstacles[count] = success_rate

        counts = list(successes_by_obstacles.keys())
        rates = list(successes_by_obstacles.values())
        ax1.bar(counts, rates, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Number of Obstacles')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate vs Obstacle Count')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Success rate by scenario difficulty
        ax2 = plt.subplot(3, 4, 2)
        difficulties = np.array([r.scenario_difficulty for r in self.results])
        successes = np.array([r.success for r in self.results])

        # Bin by difficulty
        difficulty_bins = np.linspace(difficulties.min(), difficulties.max(), 6)
        bin_indices = np.digitize(difficulties, difficulty_bins)

        bin_rates = []
        bin_centers = []
        for i in range(1, len(difficulty_bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                rate = np.mean(successes[mask]) * 100
                center = (difficulty_bins[i-1] + difficulty_bins[i]) / 2
                bin_rates.append(rate)
                bin_centers.append(center)

        ax2.bar(bin_centers, bin_rates, width=np.diff(difficulty_bins)[0]*0.8, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Scenario Difficulty (Start-Goal Distance)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate vs Scenario Difficulty')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Computation time distribution
        ax3 = plt.subplot(3, 4, 3)
        comp_times = [r.computation_time for r in self.results]
        ax3.hist(comp_times, bins=20, alpha=0.7, color='orange')
        ax3.set_xlabel('Computation Time (s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Computation Time Distribution')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Steps taken distribution
        ax4 = plt.subplot(3, 4, 4)
        successful_steps = [r.steps_taken for r in stats['successful_runs']]
        failed_steps = [r.steps_taken for r in stats['failed_runs']]

        ax4.hist(successful_steps, bins=20, alpha=0.7, label='Successful', color='green')
        ax4.hist(failed_steps, bins=20, alpha=0.7, label='Failed', color='red')
        ax4.set_xlabel('Steps Taken')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Steps Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Path efficiency (path length vs straight line distance)
        ax5 = plt.subplot(3, 4, 5)
        if stats['successful_runs']:
            straight_distances = [r.scenario_difficulty for r in stats['successful_runs']]
            path_lengths = [r.path_length for r in stats['successful_runs']]
            efficiency = np.array(straight_distances) / np.array(path_lengths) * 100

            ax5.scatter(straight_distances, efficiency, alpha=0.6, color='purple')
            ax5.set_xlabel('Straight Line Distance')
            ax5.set_ylabel('Path Efficiency (%)')
            ax5.set_title('Path Efficiency vs Distance')
            ax5.grid(True, alpha=0.3)

        # Plot 6: Failure analysis
        ax6 = plt.subplot(3, 4, 6)
        failure_types = ['Collision', 'Timeout', 'Controller Error']
        failure_counts = [
            len([r for r in stats['failed_runs'] if r.collision_occurred]),
            len([r for r in stats['failed_runs'] if r.steps_taken >= 200 and not r.collision_occurred]),
            len([r for r in stats['failed_runs'] if r.steps_taken < 200 and not r.collision_occurred])
        ]

        if sum(failure_counts) > 0:
            ax6.pie(failure_counts, labels=failure_types, autopct='%1.1f%%', startangle=90)
            ax6.set_title('Failure Type Distribution')
        else:
            ax6.text(0.5, 0.5, 'No Failures!', ha='center', va='center', transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Failure Type Distribution')

        # Plots 7-12: Sample successful and failed scenarios
        successful_examples = [r for r in stats['successful_runs']][:3]
        failed_examples = [r for r in stats['failed_runs']][:3]

        for i, (result, title_prefix) in enumerate(zip(successful_examples + failed_examples,
                                                      ['Success'] * 3 + ['Failure'] * 3)):
            ax = plt.subplot(3, 4, 7 + i)

            # This would require storing scenario info in results
            # For now, just show a summary
            ax.text(0.5, 0.5, f'{title_prefix} Example {i%3 + 1}\n'
                              f'Steps: {result.steps_taken}\n'
                              f'Obstacles: {result.num_obstacles}\n'
                              f'Distance: {result.scenario_difficulty:.2f}\n'
                              f'Time: {result.computation_time:.2f}s',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue' if 'Success' in title_prefix else 'lightcoral'))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{title_prefix} Example {i%3 + 1}')

        plt.tight_layout()
        plt.suptitle(f'MPPI Benchmark Results - Overall Success Rate: {stats["success_rate"]:.1f}%',
                    y=0.98, fontsize=16)
        plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total runs: {stats['total_runs']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Collision rate: {stats['collision_rate']:.1f}%")
        print(f"Average steps (successful): {stats['avg_steps_success']:.1f}")
        print(f"Average path length: {stats['avg_path_length']:.2f}")
        print(f"Average computation time: {stats['avg_computation_time']:.3f}s")
        print(f"Obstacle count range: {stats['obstacle_range'][0]}-{stats['obstacle_range'][1]}")
        print(f"Difficulty range: {stats['difficulty_range'][0]:.1f}-{stats['difficulty_range'][1]:.1f}")
        print("="*60)

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

    def animate_simulation(self, save_animation: bool = False):
        """Create an animation of the simulation with safety margins"""
        if not self.history:
            print("No simulation data to animate. Run simulation first.")
            return

        try:
            # Try to create animation
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title("MPPI Robot Navigation with Safety Margins")

            # Plot static elements
            for obstacle in self.obstacles:
                # Safety margin
                safety_circle = patches.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.inflated_radius,
                    color="pink",
                    alpha=0.3,
                    label="Safety Margin" if obstacle == self.obstacles[0] else ""
                )
                ax.add_patch(safety_circle)
                
                # Actual obstacle
                obstacle_circle = patches.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    color="red",
                    alpha=0.7,
                    label="Obstacle" if obstacle == self.obstacles[0] else ""
                )
                ax.add_patch(obstacle_circle)

            ax.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal')

            # Initialize dynamic elements
            robot, = ax.plot([], [], 'bo', markersize=8, label='Robot')
            trajectory, = ax.plot([], [], 'b-', alpha=0.6, linewidth=2)
            
            # Initialize safety violation indicator
            safety_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.legend()

            def animate(frame):
                if frame < len(self.history):
                    state = self.history[frame]

                    # Update robot position
                    robot.set_data([state[0]], [state[1]])

                    # Update trajectory
                    trajectory_data = np.array(self.history[:frame+1])
                    trajectory.set_data(trajectory_data[:, 0], trajectory_data[:, 1])
                    
                    # Update safety text
                    if frame < len(self.safety_violations) and self.safety_violations[frame]:
                        safety_text.set_text('SAFETY MARGIN VIOLATION!')
                        safety_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.8))
                    else:
                        safety_text.set_text('')
                        safety_text.set_bbox(dict(boxstyle='round', facecolor='white', alpha=0))

                return robot, trajectory, safety_text

            # Create animation with fallback options
            try:
                anim = FuncAnimation(fig, animate, frames=len(self.history),
                                   interval=100, blit=True, repeat=True)
            except:
                # Fallback without blitting
                anim = FuncAnimation(fig, animate, frames=len(self.history),
                                   interval=100, blit=False, repeat=True)

            if save_animation:
                try:
                    print("Saving animation... (this may take a while)")
                    anim.save('mppi_with_safety_margins.gif', writer='pillow', fps=10)
                    print("Animation saved as 'mppi_with_safety_margins.gif'")
                except Exception as e:
                    print(f"Failed to save animation: {e}")
                    print("Try installing: pip install pillow")

            plt.show()
            return anim

        except Exception as e:
            print(f"Animation failed: {e}")
            print("Creating step-by-step visualization instead...")
            self.create_step_by_step_plots()

    def create_step_by_step_plots(self):
        """Create a series of static plots showing the robot's progress with safety margins"""
        if not self.history:
            print("No simulation data available.")
            return

        trajectory = np.array(self.history)
        n_steps = len(trajectory)

        # Create 6 snapshots of the simulation
        snapshot_indices = np.linspace(0, n_steps-1, 6, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, step_idx in enumerate(snapshot_indices):
            ax = axes[i]
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Step {step_idx} / {n_steps-1}")

            # Plot obstacles with safety margins
            for obstacle in self.obstacles:
                # Safety margin
                safety_circle = patches.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.inflated_radius,
                    color="pink",
                    alpha=0.3,
                    label="Safety Margin" if obstacle == self.obstacles[0] and i == 0 else ""
                )
                ax.add_patch(safety_circle)
                
                # Actual obstacle
                obstacle_circle = patches.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    color="red",
                    alpha=0.7,
                    label="Obstacle" if obstacle == self.obstacles[0] and i == 0 else ""
                )
                ax.add_patch(obstacle_circle)

            # Plot goal
            ax.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal' if i == 0 else "")

            # Plot trajectory up to current step with safety violation highlighting
            if step_idx > 0:
                # Split into safe and violation segments
                safe_segments = []
                violation_segments = []
                
                for j in range(step_idx + 1):
                    if j < len(self.safety_violations) and self.safety_violations[j]:
                        violation_segments.append(trajectory[j])
                    else:
                        safe_segments.append(trajectory[j])
                
                # Plot connecting line
                ax.plot(trajectory[:step_idx+1, 0], trajectory[:step_idx+1, 1],
                       'k-', alpha=0.3, linewidth=1)
                
                # Plot safe segments
                if safe_segments:
                    safe_traj = np.array(safe_segments)
                    ax.scatter(
                        safe_traj[:, 0],
                        safe_traj[:, 1],
                        c="blue",
                        s=15,
                        alpha=0.8,
                        label="Safe Path" if i == 0 else "",
                    )
                
                # Plot violation segments
                if violation_segments:
                    violation_traj = np.array(violation_segments)
                    ax.scatter(
                        violation_traj[:, 0],
                        violation_traj[:, 1],
                        c="orange",
                        s=20,
                        alpha=0.9,
                        label="Safety Violations" if i == 0 else "",
                    )

            # Plot robot current position
            current_pos = trajectory[step_idx]
            ax.plot(current_pos[0], current_pos[1], 'bo', markersize=10, label='Robot' if i == 0 else "")

            # Plot velocity vector if available
            if step_idx < len(self.control_history):
                control = self.control_history[step_idx]
                ax.arrow(current_pos[0], current_pos[1],
                        control[0] * 0.5, control[1] * 0.5,
                        head_width=0.1, head_length=0.1, fc='green', ec='green')

            # Show safety status
            if step_idx < len(self.safety_violations):
                if self.safety_violations[step_idx]:
                    status_text = "SAFETY MARGIN VIOLATION!"
                    box_color = 'orange'
                else:
                    status_text = "Safe"
                    box_color = 'lightgreen'
                
                ax.text(0.02, 0.98, status_text, transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

            if i == 0:
                ax.legend()

        plt.tight_layout()
        plt.suptitle("MPPI Robot Navigation with Safety Margins - Step by Step Progress", y=1.02, fontsize=16)
        plt.show()

    def create_interactive_plot(self):
        """Create an interactive plot that you can step through manually"""
        if not self.history:
            print("No simulation data available.")
            return

        trajectory = np.array(self.history)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Plot obstacles
        for obstacle in self.obstacles:
            circle = patches.Circle((obstacle.x, obstacle.y), obstacle.radius,
                                  color='red', alpha=0.7, label='Obstacle' if obstacle == self.obstacles[0] else "")
            ax.add_patch(circle)

        # Plot goal
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal')

        # Plot full trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.4, linewidth=1, label='Full Path')

        # Plot start and end positions
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=12, label='End')

        # Add some key waypoints
        n_waypoints = min(10, len(trajectory))
        waypoint_indices = np.linspace(0, len(trajectory)-1, n_waypoints, dtype=int)

        for i, idx in enumerate(waypoint_indices):
            ax.plot(trajectory[idx, 0], trajectory[idx, 1], 'ko', markersize=6, alpha=0.7)
            if i % 2 == 0:  # Label every other waypoint to avoid clutter
                ax.annotate(f't={idx}', (trajectory[idx, 0], trajectory[idx, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.legend()
        ax.set_title("MPPI Robot Navigation - Complete Trajectory")
        plt.show()

def main():
    """Main function with options for single simulation or benchmark"""
    print("=" * 60)
    print("MPPI Controller with Inflated Obstacle Safety Margins")
    print("ME/CDS/EE 234(b) Final Project Simulation")
    print("=" * 60)
    
    print("\nParameters:")
    print("- Horizon: 20")
    print("- Number of samples: 100")
    print("- Lambda (temperature): 1.0")
    print("- Sigma (control noise): 1.0")
    print("- Safety margin: 0.2")
    print("=" * 60)

    # Ask user for simulation type
    print("\nChoose simulation type:")
    print("1. Single scenario (with safety margins)")
    print("2. Benchmark with random scenarios (with safety margins)")
    print("3. Quick benchmark (5 scenarios, 2 trials each)")
    print("4. Comprehensive benchmark (20 scenarios, 3 trials each)")

    try:
        choice = input("Enter choice (1-4): ").strip()
    except:
        choice = "1"  # Default to single scenario

    if choice == "1":
        # Original single scenario simulation
        run_single_scenario()

    elif choice == "2":
        # Custom benchmark
        try:
            num_scenarios = int(input("Number of scenarios (default 10): ") or "10")
            trials_per_scenario = int(input("Trials per scenario (default 2): ") or "2")
        except:
            num_scenarios, trials_per_scenario = 10, 2

        run_benchmark(num_scenarios, trials_per_scenario)

    elif choice == "3":
        # Quick benchmark
        print("Running quick benchmark...")
        run_benchmark(5, 2)

    elif choice == "4":
        # Comprehensive benchmark
        print("Running comprehensive benchmark...")
        run_benchmark(20, 3)

    else:
        print("Invalid choice, running single scenario...")
        run_single_scenario()

def run_single_scenario():
    """Run the original single scenario simulation with safety margins"""
    print("\nRunning single scenario simulation with safety margins...")

    # Create and run simulation with safety margin
    safety_margin = 0.2
    sim = MPPISimulation(safety_margin=safety_margin)

    # Run the simulation
    start_time = time.time()
    success = sim.run_simulation(max_steps=500)
    end_time = time.time()

    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {success}")

    if sim.history:
        final_state = sim.history[-1]
        final_distance = np.linalg.norm(final_state[:2] - sim.goal)
        print(f"Final distance to goal: {final_distance:.3f}")
        print(f"Total path length: {sum(np.linalg.norm(np.diff(np.array(sim.history)[:, :2], axis=0), axis=1)):.3f}")
        print(f"Safety violations: {sum(sim.safety_violations)} out of {len(sim.safety_violations)} steps")

    # Plot results
    sim.plot_results()

    # Try different visualization options
    print("\nGenerating visualizations...")

    # Try animation first, with fallback options
    try:
        print("Attempting to create animation...")
        anim = sim.animate_simulation(save_animation=False)
    except Exception as e:
        print(f"Animation failed: {e}")

    # Always create step-by-step plots as backup
    print("Creating step-by-step visualization...")
    sim.create_step_by_step_plots()

    # Create interactive plot
    print("Creating interactive trajectory plot...")
    sim.create_interactive_plot()

    # Performance analysis
    if sim.cost_history:
        print(f"\nPerformance Analysis:")
        print(f"Initial cost: {sim.cost_history[0]:.2f}")
        print(f"Final cost: {sim.cost_history[-1]:.2f}")
        print(f"Cost reduction: {(sim.cost_history[0] - sim.cost_history[-1]) / sim.cost_history[0] * 100:.1f}%")

def run_benchmark(num_scenarios: int, trials_per_scenario: int):
    """Run benchmark with random scenarios"""
    print(f"\nRunning benchmark with {num_scenarios} scenarios, {trials_per_scenario} trials each...")

    # Create benchmark suite
    benchmark = MPPIBenchmark()

    # Run benchmark
    start_time = time.time()
    results = benchmark.run_benchmark(
        num_scenarios=num_scenarios,
        trials_per_scenario=trials_per_scenario,
        max_steps=200,
        verbose=True
    )
    end_time = time.time()

    print(f"\nBenchmark completed in {end_time - start_time:.2f} seconds")

    # Analyze and plot results
    stats = benchmark.analyze_results()
    benchmark.plot_benchmark_results()

    # Additional analysis
    print("\nDetailed Analysis:")
    if stats['successful_runs']:
        successful_steps = [r.steps_taken for r in stats['successful_runs']]
        print(f"Steps for successful runs: {np.mean(successful_steps):.1f} ± {np.std(successful_steps):.1f}")

        path_lengths = [r.path_length for r in stats['successful_runs']]
        straight_distances = [r.scenario_difficulty for r in stats['successful_runs']]
        efficiencies = np.array(straight_distances) / np.array(path_lengths) * 100
        print(f"Path efficiency: {np.mean(efficiencies):.1f}% ± {np.std(efficiencies):.1f}%")

    # Success rate by difficulty
    all_difficulties = [r.scenario_difficulty for r in results]
    all_successes = [r.success for r in results]

    # Divide into easy, medium, hard scenarios
    difficulty_thirds = np.percentile(all_difficulties, [33, 67])
    easy_mask = np.array(all_difficulties) <= difficulty_thirds[0]
    medium_mask = (np.array(all_difficulties) > difficulty_thirds[0]) & (np.array(all_difficulties) <= difficulty_thirds[1])
    hard_mask = np.array(all_difficulties) > difficulty_thirds[1]

    easy_success = np.mean(np.array(all_successes)[easy_mask]) * 100
    medium_success = np.mean(np.array(all_successes)[medium_mask]) * 100
    hard_success = np.mean(np.array(all_successes)[hard_mask]) * 100

    print(f"\nSuccess rates by difficulty:")
    print(f"Easy scenarios (≤{difficulty_thirds[0]:.1f}): {easy_success:.1f}%")
    print(f"Medium scenarios ({difficulty_thirds[0]:.1f}-{difficulty_thirds[1]:.1f}): {medium_success:.1f}%")
    print(f"Hard scenarios (>{difficulty_thirds[1]:.1f}): {hard_success:.1f}%")

    # Show some example scenarios
    print("\nExample successful scenario:")
    if stats['successful_runs']:
        example = stats['successful_runs'][0]
        print(f"  Steps: {example.steps_taken}, Obstacles: {example.num_obstacles}")
        print(f"  Distance: {example.scenario_difficulty:.2f}, Path length: {example.path_length:.2f}")

    print("\nExample failed scenario:")
    if stats['failed_runs']:
        example = stats['failed_runs'][0]
        print(f"  Steps: {example.steps_taken}, Obstacles: {example.num_obstacles}")
        print(f"  Distance: {example.scenario_difficulty:.2f}, Collision: {example.collision_occurred}")

    return results, stats

def demonstrate_random_scenarios():
    """Demonstrate a few random scenarios visually"""
    print("Demonstrating random scenario generation...")

    env_gen = RandomEnvironmentGenerator(safety_margin=0.2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]

        # Generate random scenario
        start_state, goal, obstacles = env_gen.generate_random_scenario(seed=i)

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

        # Plot straight line
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

if __name__ == "__main__":
    # Comment out the automatic demonstration of random scenarios
    demonstrate_random_scenarios()
    
    main()