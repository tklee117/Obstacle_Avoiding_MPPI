import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import time
import pandas as pd
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ===== ORIGINAL MPPI CODE =====
@dataclass
class Obstacle:
    """Represents a circular obstacle in the environment"""
    x: float
    y: float
    radius: float
    safety_margin: float = 0.7  # Safety margin to be added to radius

    @property
    def inflated_radius(self) -> float:
        """Return the radius including safety margin"""
        return self.radius + self.safety_margin


class MPPIController:
    """Model Predictive Path Integral Controller with Inflated Obstacle Safety"""

    def __init__(
        self,
        horizon: int = 20,
        num_samples: int = 2000,
        control_dim: int = 2,
        state_dim: int = 4,
        dt: float = 0.1,
        lambda_: float = 1.0,
        sigma: float = 0.5,
        control_bounds: Tuple[float, float] = (-2.0, 2.0),
        safety_margin: float = 0.7,
    ):
        """Initialize MPPI controller"""
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

        # Cost weights - adjusted for inflated obstacles
        self.Q_goal = 50.0  # Goal reaching weight
        self.Q_obstacle = 500.0  # Obstacle avoidance weight
        self.R = 0.1  # Control effort weight
        self.Q_velocity = 0.5  # Velocity regulation weight

        # Storage for visualization
        self.last_sampled_trajectories = []
        self.last_weights = []
        self.last_costs = []

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Simple 2D point mass dynamics: double integrator"""
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
        """Rollout a trajectory given initial state and control sequence"""
        trajectory = np.zeros((self.horizon + 1, self.state_dim))
        trajectory[0] = initial_state

        for t in range(self.horizon):
            trajectory[t + 1] = self.dynamics(trajectory[t], control_sequence[t])

        return trajectory

    def compute_cost(self, trajectory: np.ndarray, goal: np.ndarray, 
                    obstacles: List[Obstacle], control_sequence: np.ndarray) -> float:
        """Compute the cost of a trajectory using inflated obstacles"""
        cost = 0.0
        collision_penalty = 0.0

        # Terminal cost: distance to goal
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

            # Obstacle avoidance cost using inflated obstacles
            pos = state[:2]
            for obstacle in obstacles:
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                distance = np.linalg.norm(pos - obstacle_pos)

                # Use inflated radius as the hard boundary
                inflated_radius = obstacle.inflated_radius

                # Check if inside the inflated obstacle (safety zone)
                if distance < inflated_radius:
                    # Penetration into safety zone
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

    def sample_controls(self) -> np.ndarray:
        """Sample control sequences around the current control sequence"""
        # Generate random noise
        noise = np.random.normal(0, self.sigma, (self.num_samples, self.horizon, self.control_dim))

        # Add noise to current control sequence
        sampled_controls = self.U[None, :, :] + noise

        # Clip controls to bounds
        sampled_controls = np.clip(sampled_controls, self.control_bounds[0], self.control_bounds[1])

        return sampled_controls

    def update_control(self, state: np.ndarray, goal: np.ndarray, 
                      obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update the control sequence using MPPI with inflated obstacles"""
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

        # Store for visualization
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
            weights = np.ones(self.num_samples) / self.num_samples
        else:
            weights /= weights_sum

        # Check for NaN in weights
        if np.any(np.isnan(weights)):
            weights = np.ones(self.num_samples) / self.num_samples

        # Store weights for visualization
        self.last_weights = weights.copy()

        # Weighted average of control sequences
        self.U = np.sum(weights[:, None, None] * sampled_controls, axis=0)

        # Check for NaN in control update
        if np.any(np.isnan(self.U)):
            self.U += np.random.normal(0, 0.1, self.U.shape)

        # Shift control sequence for next iteration (warm start)
        self.U[:-1] = self.U[1:].copy()
        self.U[-1] = np.zeros(self.control_dim)

        # Return best trajectory for visualization
        best_idx = np.argmin(costs)
        best_trajectory = trajectories[best_idx]

        return self.U[0], best_trajectory, costs


class MPPISimulation:
    """Simulation environment for MPPI with inflated obstacle safety margins"""

    def __init__(self, safety_margin: float = 0.7):
        self.safety_margin = safety_margin
        self.controller = MPPIController(
            horizon=15,
            num_samples=2000,
            dt=0.1,
            lambda_=10.0,
            sigma=10.0,
            safety_margin=safety_margin,
        )

        # Environment setup
        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # Start at origin
        self.goal = np.array([8.0, 8.0])  # Goal position

        # Create obstacles with consistent safety margin
        self.obstacles = [
            Obstacle(2.5, 2.0, 0.6, safety_margin),
            Obstacle(5.0, 4.0, 0.8, safety_margin),
            Obstacle(3.0, 6.0, 0.5, safety_margin),
            Obstacle(6.5, 2.5, 0.7, safety_margin),
            Obstacle(6.0, 6.5, 0.6, safety_margin),
        ]

        # Simulation history
        self.history = []
        self.control_history = []
        self.cost_history = []
        self.rollout_history = []
        self.safety_violations = []  # Track when robot enters safety margins

    def run_simulation(self, max_steps: int = 200) -> bool:
        """Run the simulation with inflated obstacle safety checking"""
        
        for step in range(max_steps):
            # Get control from MPPI
            control, best_trajectory, costs = self.controller.update_control(
                self.state, self.goal, self.obstacles
            )

            # Store rollout data for visualization
            self.rollout_history.append({
                'trajectories': [traj.copy() for traj in self.controller.last_sampled_trajectories],
                'weights': self.controller.last_weights.copy(),
                'costs': self.controller.last_costs.copy(),
                'current_state': self.state.copy(),
            })

            # Apply control and update state
            self.state = self.controller.dynamics(self.state, control)

            # Check safety violations (robot in safety margin but not colliding)
            safety_violation = False
            actual_collision = False

            for obstacle in self.obstacles:
                distance = np.linalg.norm(self.state[:2] - np.array([obstacle.x, obstacle.y]))

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
                return True

            # Check actual collision (not just safety margin)
            if actual_collision:
                return False

        return False

# ===== TESTING FRAMEWORK =====

@dataclass
class TestResult:
    """Store results from a single test run"""
    # Environment parameters
    num_obstacles: int
    obstacle_density: float
    environment_complexity: float
    
    # Algorithm parameters
    num_samples: int
    sigma: float
    lambda_: float
    horizon: int
    safety_margin: float
    Q_goal: float
    Q_obstacle: float
    R: float
    
    # Performance metrics
    success: bool
    path_length: float
    computation_time: float
    safety_violations: int
    safety_violation_rate: float
    min_clearance: float
    goal_distance: float
    control_effort: float
    path_smoothness: float
    convergence_time: float
    
    # Additional metrics
    effective_samples_mean: float
    cost_reduction: float
    num_iterations: int

class EnvironmentGenerator:
    """Generate randomized test environments"""
    
    def __init__(self, workspace_bounds: Tuple[float, float, float, float] = (0, 10, 0, 10)):
        self.x_min, self.x_max, self.y_min, self.y_max = workspace_bounds
        self.workspace_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def generate_random_environment(self, 
                                  num_obstacles: Optional[int] = None,
                                  min_obstacles: int = 2,
                                  max_obstacles: int = 15,
                                  min_radius: float = 0.3,
                                  max_radius: float = 1.0,
                                  start_goal_min_distance: float = 5.0,
                                  obstacle_clearance: float = 0.5) -> Dict:
        """Generate a random environment with obstacles, start, and goal"""
        
        if num_obstacles is None:
            num_obstacles = np.random.randint(min_obstacles, max_obstacles + 1)
        
        # Generate start and goal positions
        while True:
            start = np.array([
                np.random.uniform(self.x_min + 1, self.x_max - 1),
                np.random.uniform(self.y_min + 1, self.y_max - 1)
            ])
            goal = np.array([
                np.random.uniform(self.x_min + 1, self.x_max - 1),
                np.random.uniform(self.y_min + 1, self.y_max - 1)
            ])
            
            if np.linalg.norm(goal - start) >= start_goal_min_distance:
                break
        
        # Generate obstacles
        obstacles = []
        attempts = 0
        max_attempts = 1000
        
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Random position and radius
            x = np.random.uniform(self.x_min + max_radius, self.x_max - max_radius)
            y = np.random.uniform(self.y_min + max_radius, self.y_max - max_radius)
            radius = np.random.uniform(min_radius, max_radius)
            
            pos = np.array([x, y])
            
            # Check clearance from start and goal
            if (np.linalg.norm(pos - start) < radius + obstacle_clearance or
                np.linalg.norm(pos - goal) < radius + obstacle_clearance):
                continue
            
            # Check overlap with existing obstacles
            valid = True
            for existing_obs in obstacles:
                existing_pos = np.array([existing_obs.x, existing_obs.y])
                if np.linalg.norm(pos - existing_pos) < radius + existing_obs.radius + obstacle_clearance:
                    valid = False
                    break
            
            if valid:
                obstacles.append(Obstacle(x, y, radius))
        
        # Calculate environment complexity metrics
        total_obstacle_area = sum(np.pi * obs.radius**2 for obs in obstacles)
        obstacle_density = total_obstacle_area / self.workspace_area
        
        # Complexity based on obstacle density and spatial distribution
        if len(obstacles) > 1:
            positions = np.array([[obs.x, obs.y] for obs in obstacles])
            mean_separation = np.mean([np.min([np.linalg.norm(positions[i] - positions[j]) 
                                             for j in range(len(positions)) if i != j])
                                     for i in range(len(positions))])
            complexity = obstacle_density * len(obstacles) / max(mean_separation, 0.1)
        else:
            complexity = obstacle_density
        
        return {
            'start': start,
            'goal': goal,
            'obstacles': obstacles,
            'num_obstacles': len(obstacles),
            'obstacle_density': obstacle_density,
            'environment_complexity': complexity
        }

class MPPITester:
    """Comprehensive testing framework for MPPI algorithm"""
    
    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self.env_generator = EnvironmentGenerator()
        self.results = []
    
    def run_single_test(self, 
                       environment: Dict,
                       algorithm_params: Dict,
                       safety_margin: float = 0.7,
                       verbose: bool = False) -> TestResult:
        """Run a single test with given environment and parameters"""
        
        # Create MPPI controller with specified parameters
        controller = MPPIController(
            horizon=algorithm_params.get('horizon', 15),
            num_samples=algorithm_params.get('num_samples', 2000),
            lambda_=algorithm_params.get('lambda_', 10.0),
            sigma=algorithm_params.get('sigma', 10.0),
            safety_margin=safety_margin
        )
        
        # Update cost weights if specified
        if 'Q_goal' in algorithm_params:
            controller.Q_goal = algorithm_params['Q_goal']
        if 'Q_obstacle' in algorithm_params:
            controller.Q_obstacle = algorithm_params['Q_obstacle']
        if 'R' in algorithm_params:
            controller.R = algorithm_params['R']
        
        # Create simulation
        sim = MPPISimulation(safety_margin=safety_margin)
        sim.controller = controller
        sim.state = np.array([environment['start'][0], environment['start'][1], 0.0, 0.0])
        sim.goal = environment['goal']
        sim.obstacles = environment['obstacles']
        
        # Clear previous history
        sim.history = []
        sim.control_history = []
        sim.cost_history = []
        sim.safety_violations = []
        sim.rollout_history = []
        
        # Run simulation and measure time
        start_time = time.time()
        try:
            success = sim.run_simulation(max_steps=self.max_steps)
        except Exception as e:
            if verbose:
                print(f"Simulation failed: {e}")
            success = False
        end_time = time.time()
        
        # Calculate metrics
        computation_time = end_time - start_time
        num_iterations = len(sim.history)
        
        if sim.history:
            # Path metrics
            trajectory = np.array(sim.history)
            path_length = self.calculate_path_length(trajectory)
            
            # Safety metrics
            safety_violations = sum(sim.safety_violations)
            safety_violation_rate = safety_violations / len(sim.safety_violations) if sim.safety_violations else 0
            min_clearance = self.calculate_min_clearance(trajectory, environment['obstacles'])
            
            # Goal distance
            final_state = sim.history[-1]
            goal_distance = np.linalg.norm(final_state[:2] - environment['goal'])
            
            # Control effort
            if sim.control_history:
                controls = np.array(sim.control_history)
                control_effort = np.sum(np.linalg.norm(controls, axis=1))
            else:
                control_effort = 0.0
            
            # Path smoothness (total variation in velocity)
            if len(trajectory) > 1:
                velocity_changes = np.diff(trajectory[:, 2:4], axis=0)
                path_smoothness = np.sum(np.linalg.norm(velocity_changes, axis=1))
            else:
                path_smoothness = 0.0
            
            # Convergence time (time to get within 1.0 of goal)
            convergence_time = num_iterations
            for i, state in enumerate(trajectory):
                if np.linalg.norm(state[:2] - environment['goal']) < 1.0:
                    convergence_time = i
                    break
            
            # Cost reduction
            if sim.cost_history and len(sim.cost_history) > 1:
                cost_reduction = (sim.cost_history[0] - sim.cost_history[-1]) / max(sim.cost_history[0], 1e-6)
            else:
                cost_reduction = 0.0
            
            # Effective samples (average over all iterations)
            if hasattr(sim, 'rollout_history') and sim.rollout_history:
                effective_samples = []
                for rollout_data in sim.rollout_history:
                    if 'weights' in rollout_data:
                        weights = rollout_data['weights']
                        eff_samples = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0
                        effective_samples.append(eff_samples)
                effective_samples_mean = np.mean(effective_samples) if effective_samples else 0
            else:
                effective_samples_mean = 0
        else:
            # No trajectory generated
            path_length = float('inf')
            safety_violations = 0
            safety_violation_rate = 0
            min_clearance = 0
            goal_distance = np.linalg.norm(environment['start'] - environment['goal'])
            control_effort = 0
            path_smoothness = 0
            convergence_time = self.max_steps
            cost_reduction = 0
            effective_samples_mean = 0
        
        # Create test result
        result = TestResult(
            # Environment
            num_obstacles=environment['num_obstacles'],
            obstacle_density=environment['obstacle_density'],
            environment_complexity=environment['environment_complexity'],
            
            # Algorithm parameters
            num_samples=algorithm_params.get('num_samples', 2000),
            sigma=algorithm_params.get('sigma', 10.0),
            lambda_=algorithm_params.get('lambda_', 10.0),
            horizon=algorithm_params.get('horizon', 15),
            safety_margin=safety_margin,
            Q_goal=controller.Q_goal,
            Q_obstacle=controller.Q_obstacle,
            R=controller.R,
            
            # Performance metrics
            success=success,
            path_length=path_length,
            computation_time=computation_time,
            safety_violations=safety_violations,
            safety_violation_rate=safety_violation_rate,
            min_clearance=min_clearance,
            goal_distance=goal_distance,
            control_effort=control_effort,
            path_smoothness=path_smoothness,
            convergence_time=convergence_time,
            effective_samples_mean=effective_samples_mean,
            cost_reduction=cost_reduction,
            num_iterations=num_iterations
        )
        
        if verbose:
            print(f"Test completed: Success={success}, Path Length={path_length:.2f}, "
                  f"Safety Violations={safety_violations}, Time={computation_time:.2f}s")
        
        return result
    
    def calculate_path_length(self, trajectory: np.ndarray) -> float:
        """Calculate total path length"""
        if len(trajectory) < 2:
            return 0.0
        diffs = np.diff(trajectory[:, :2], axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def calculate_min_clearance(self, trajectory: np.ndarray, obstacles: List) -> float:
        """Calculate minimum clearance from obstacles during trajectory"""
        min_clearance = float('inf')
        for state in trajectory:
            pos = state[:2]
            for obstacle in obstacles:
                obs_pos = np.array([obstacle.x, obstacle.y])
                clearance = np.linalg.norm(pos - obs_pos) - obstacle.radius
                min_clearance = min(min_clearance, clearance)
        return max(0, min_clearance)

def run_extensive_parameter_testing():
    """Run extensive parameter testing with comprehensive parameter ranges"""
    
    print("="*60)
    print("EXTENSIVE MPPI PARAMETER TESTING")
    print("="*60)
    
    tester = MPPITester(max_steps=200)
    
    # Comprehensive parameter ranges
    print("\nParameter ranges being tested:")
    
    # Sigma values 
    sigma_values = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
    print(f"Sigma values: {sigma_values}")
    
    # Lambda values
    lambda_values = [0.5, 1.0, 5.0]
    print(f"Lambda values: {lambda_values}")
    
    # Number of samples
    num_samples_values = [500, 1000, 2000, 4000]
    print(f"Number of samples: {num_samples_values}")
    
    # Horizon lengths
    horizon_values = [10, 20, 30]
    print(f"Horizon values: {horizon_values}")
    
    # Safety margins
    safety_margins = [0.1, 0.3, 0.5]
    print(f"Safety margins: {safety_margins}")
    
    # Number of trials per configuration
    trials_per_config = 5
    environments_per_trial = 3
    
    total_configs = len(sigma_values) * len(lambda_values) * len(num_samples_values) * len(horizon_values) * len(safety_margins)
    total_tests = total_configs * trials_per_config * environments_per_trial
    
    print(f"\nTotal configurations: {total_configs}")
    print(f"Total tests: {total_tests}")
    print(f"Estimated time: {total_tests * 2 / 60:.1f} minutes (assuming 2s per test)")
    
    input("\nPress Enter to start extensive testing...")
    
    all_results = []
    test_count = 0
    start_time = time.time()
    
    # Main testing loop
    for sigma in sigma_values:
        for lambda_ in lambda_values:
            for num_samples in num_samples_values:
                for horizon in horizon_values:
                    for safety_margin in safety_margins:
                        
                        config = {
                            'num_samples': num_samples,
                            'sigma': sigma,
                            'lambda_': lambda_,
                            'horizon': horizon
                        }
                        
                        print(f"\nTesting config: σ={sigma}, λ={lambda_}, samples={num_samples}, "
                              f"horizon={horizon}, safety={safety_margin}")
                        
                        config_results = []
                        
                        for trial in range(trials_per_config):
                            for env_trial in range(environments_per_trial):
                                test_count += 1
                                
                                if test_count % 50 == 0:
                                    elapsed = time.time() - start_time
                                    rate = test_count / elapsed
                                    eta = (total_tests - test_count) / rate / 60
                                    print(f"  Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.1f}%) "
                                          f"- ETA: {eta:.1f} min")
                                
                                try:
                                    # Generate random environment
                                    environment = tester.env_generator.generate_random_environment(
                                        min_obstacles=3, 
                                        max_obstacles=12,
                                        min_radius=0.3,
                                        max_radius=0.9
                                    )
                                    
                                    # Run test
                                    result = tester.run_single_test(
                                        environment=environment,
                                        algorithm_params=config,
                                        safety_margin=safety_margin,
                                        verbose=False
                                    )
                                    
                                    config_results.append(result)
                                    all_results.append(result)
                                    
                                except Exception as e:
                                    print(f"    Test {test_count} failed: {e}")
                                    continue
                        
                        # Print summary for this configuration
                        if config_results:
                            success_rate = sum(r.success for r in config_results) / len(config_results)
                            avg_time = sum(r.computation_time for r in config_results) / len(config_results)
                            print(f"  Config summary: Success={success_rate:.3f}, AvgTime={avg_time:.3f}s")
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(result) for result in all_results])
    
    print(f"\n" + "="*60)
    print("EXTENSIVE TESTING COMPLETE!")
    print("="*60)
    print(f"Total tests completed: {len(all_results)}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    return df

def analyze_extensive_results(df):
    """Comprehensive analysis of extensive test results"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS OF EXTENSIVE TESTING RESULTS")
    print("="*60)
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Total tests: {len(df)}")
    print(f"Overall success rate: {df['success'].mean():.3f}")
    print(f"Mean computation time: {df['computation_time'].mean():.3f}s")
    print(f"Mean safety violation rate: {df['safety_violation_rate'].mean():.3f}")
    
    # Parameter sensitivity analysis
    print(f"\nPARAMETER SENSITIVITY ANALYSIS:")
    
    # Sigma analysis
    print(f"\n1. SIGMA ANALYSIS:")
    sigma_stats = df.groupby('sigma').agg({
        'success': ['mean', 'std', 'count'],
        'path_length': lambda x: x[x < np.inf].mean() if len(x[x < np.inf]) > 0 else np.nan,
        'computation_time': 'mean',
        'safety_violation_rate': 'mean'
    }).round(3)
    print(sigma_stats)
    
    # Lambda analysis
    print(f"\n2. LAMBDA ANALYSIS:")
    lambda_stats = df.groupby('lambda_').agg({
        'success': ['mean', 'std', 'count'],
        'path_length': lambda x: x[x < np.inf].mean() if len(x[x < np.inf]) > 0 else np.nan,
        'computation_time': 'mean',
        'safety_violation_rate': 'mean'
    }).round(3)
    print(lambda_stats)
    
    # Number of samples analysis
    print(f"\n3. NUMBER OF SAMPLES ANALYSIS:")
    samples_stats = df.groupby('num_samples').agg({
        'success': ['mean', 'std', 'count'],
        'path_length': lambda x: x[x < np.inf].mean() if len(x[x < np.inf]) > 0 else np.nan,
        'computation_time': 'mean',
        'safety_violation_rate': 'mean'
    }).round(3)
    print(samples_stats)
    
    # Create comprehensive visualizations
    plt.style.use('default')
    
    # Figure 1: Parameter sensitivity plots
    fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig1.suptitle('MPPI Parameter Sensitivity Analysis', fontsize=16)
    
    # Sigma vs Success Rate
    sigma_success = df.groupby('sigma')['success'].agg(['mean', 'std']).reset_index()
    axes[0,0].errorbar(sigma_success['sigma'], sigma_success['mean'], 
                       yerr=sigma_success['std'], marker='o', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Sigma (Control Noise)')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].set_title('Success Rate vs Sigma')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xscale('log')
    
    # Lambda vs Success Rate
    lambda_success = df.groupby('lambda_')['success'].agg(['mean', 'std']).reset_index()
    axes[0,1].errorbar(lambda_success['lambda_'], lambda_success['mean'], 
                       yerr=lambda_success['std'], marker='s', linewidth=2, markersize=8, color='orange')
    axes[0,1].set_xlabel('Lambda (Temperature)')
    axes[0,1].set_ylabel('Success Rate')
    axes[0,1].set_title('Success Rate vs Lambda')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xscale('log')
    
    # Samples vs Success Rate
    samples_success = df.groupby('num_samples')['success'].agg(['mean', 'std']).reset_index()
    axes[0,2].errorbar(samples_success['num_samples'], samples_success['mean'], 
                       yerr=samples_success['std'], marker='^', linewidth=2, markersize=8, color='green')
    axes[0,2].set_xlabel('Number of Samples')
    axes[0,2].set_ylabel('Success Rate')
    axes[0,2].set_title('Success Rate vs Number of Samples')
    axes[0,2].grid(True, alpha=0.3)
    
    # Sigma vs Computation Time
    sigma_time = df.groupby('sigma')['computation_time'].agg(['mean', 'std']).reset_index()
    axes[1,0].errorbar(sigma_time['sigma'], sigma_time['mean'], 
                       yerr=sigma_time['std'], marker='o', linewidth=2, markersize=8, color='red')
    axes[1,0].set_xlabel('Sigma (Control Noise)')
    axes[1,0].set_ylabel('Computation Time (s)')
    axes[1,0].set_title('Computation Time vs Sigma')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xscale('log')
    
    # Lambda vs Path Length (successful runs only)
    success_df = df[df['success'] == True]
    if len(success_df) > 0:
        lambda_path = success_df.groupby('lambda_')['path_length'].agg(['mean', 'std']).reset_index()
        axes[1,1].errorbar(lambda_path['lambda_'], lambda_path['mean'], 
                           yerr=lambda_path['std'], marker='s', linewidth=2, markersize=8, color='purple')
        axes[1,1].set_xlabel('Lambda (Temperature)')
        axes[1,1].set_ylabel('Path Length')
        axes[1,1].set_title('Path Length vs Lambda (Successful Runs)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xscale('log')
    
    # Safety Margin vs Safety Violations
    safety_violations = df.groupby('safety_margin')['safety_violation_rate'].agg(['mean', 'std']).reset_index()
    axes[1,2].errorbar(safety_violations['safety_margin'], safety_violations['mean'], 
                       yerr=safety_violations['std'], marker='^', linewidth=2, markersize=8, color='brown')
    axes[1,2].set_xlabel('Safety Margin')
    axes[1,2].set_ylabel('Safety Violation Rate')
    axes[1,2].set_title('Safety Violations vs Safety Margin')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Heatmaps
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('MPPI Parameter Interaction Heatmaps', fontsize=16)
    
    # Success rate heatmap: Sigma vs Lambda
    pivot_success = df.pivot_table(values='success', index='sigma', columns='lambda_', aggfunc='mean')
    if not pivot_success.empty:
        sns.heatmap(pivot_success, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,0])
        axes[0,0].set_title('Success Rate: Sigma vs Lambda')
    
    # Computation time heatmap: Samples vs Horizon
    pivot_time = df.pivot_table(values='computation_time', index='num_samples', columns='horizon', aggfunc='mean')
    if not pivot_time.empty:
        sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,1])
        axes[0,1].set_title('Computation Time: Samples vs Horizon')
    
    # Path length heatmap: Sigma vs Samples (successful runs)
    if len(success_df) > 0:
        pivot_path = success_df.pivot_table(values='path_length', index='sigma', columns='num_samples', aggfunc='mean')
        if not pivot_path.empty:
            sns.heatmap(pivot_path, annot=True, fmt='.1f', cmap='viridis', ax=axes[1,0])
            axes[1,0].set_title('Path Length: Sigma vs Samples (Success Only)')
    
    # Safety violations: Safety Margin vs Sigma
    pivot_safety = df.pivot_table(values='safety_violation_rate', index='safety_margin', columns='sigma', aggfunc='mean')
    if not pivot_safety.empty:
        sns.heatmap(pivot_safety, annot=True, fmt='.2f', cmap='Reds', ax=axes[1,1])
        axes[1,1].set_title('Safety Violations: Safety Margin vs Sigma')
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal configurations
    print(f"\n" + "="*50)
    print("OPTIMAL CONFIGURATION ANALYSIS")
    print("="*50)
    
    # Best overall performance (highest success rate)
    best_success = df.groupby(['sigma', 'lambda_', 'num_samples', 'horizon']).agg({
        'success': 'mean',
        'path_length': lambda x: x[x < np.inf].mean() if len(x[x < np.inf]) > 0 else np.nan,
        'computation_time': 'mean',
        'safety_violation_rate': 'mean'
    }).reset_index()
    
    best_success_configs = best_success.nlargest(10, 'success')
    print("\nTop 10 configurations by success rate:")
    print(best_success_configs)
    
    # Best efficiency (success rate / computation time)
    best_success['efficiency'] = best_success['success'] / best_success['computation_time']
    best_efficiency_configs = best_success.nlargest(10, 'efficiency')
    print("\nTop 10 configurations by efficiency (success/time):")
    print(best_efficiency_configs[['sigma', 'lambda_', 'num_samples', 'horizon', 'success', 'computation_time', 'efficiency']])
    
    return df

# Usage instructions
if __name__ == "__main__":
    print("EXTENSIVE MPPI TESTING FRAMEWORK")
    print("=================================")
    print()
    print("This will run comprehensive parameter testing including:")
    print("- Sigma values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]")
    print("- Lambda values: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]")
    print("- Sample counts: [500, 1000, 1500, 2000, 3000, 4000]")
    print("- Horizons: [10, 15, 20, 25, 30]")
    print("- Safety margins: [0.3, 0.5, 0.7, 1.0]")
    print()
    print("Estimated total tests: ~21,000")
    print("Estimated time: 10-20 hours (depending on your machine)")
    print()
    
    choice = input("Choose option:\n1. Run extensive testing\n2. Run medium testing (fewer parameters)\n3. Exit\nChoice (1/2/3): ")
    
    if choice == "1":
        # Run full extensive testing
        df = run_extensive_parameter_testing()
        analyze_extensive_results(df)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"mppi_extensive_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
    elif choice == "2":
        # Run medium-scale testing
        print("\nRunning medium-scale testing...")
        
        tester = MPPITester(max_steps=200)
        
        # Reduced parameter ranges
        sigma_values = [0.1, 1.0, 10.0]
        lambda_values = [1.0, 5.0, 20.0]
        num_samples_values = [10, 100, 1000]
        horizon_values = [20]
        safety_margins = [0.2, 0.5]
        
        print(f"Testing {len(sigma_values)} sigma values, {len(lambda_values)} lambda values,")
        print(f"{len(num_samples_values)} sample counts, {len(horizon_values)} horizons,")
        print(f"{len(safety_margins)} safety margins")
        
        total_configs = len(sigma_values) * len(lambda_values) * len(num_samples_values) * len(horizon_values) * len(safety_margins)
        total_tests = total_configs * 3  # 3 trials per config
        print(f"Total tests: {total_tests}")
        
        input("Press Enter to start medium testing...")
        
        all_results = []
        test_count = 0
        start_time = time.time()
        
        for sigma in sigma_values:
            for lambda_ in lambda_values:
                for num_samples in num_samples_values:
                    for horizon in horizon_values:
                        for safety_margin in safety_margins:
                            
                            config = {
                                'num_samples': num_samples,
                                'sigma': sigma,
                                'lambda_': lambda_,
                                'horizon': horizon
                            }
                            
                            print(f"\nTesting: σ={sigma}, λ={lambda_}, samples={num_samples}, "
                                  f"horizon={horizon}, safety={safety_margin}")
                            
                            # 3 trials per configuration
                            for trial in range(3):
                                test_count += 1
                                
                                if test_count % 20 == 0:
                                    elapsed = time.time() - start_time
                                    rate = test_count / elapsed
                                    eta = (total_tests - test_count) / rate / 60
                                    print(f"  Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.1f}%) "
                                          f"- ETA: {eta:.1f} min")
                                
                                try:
                                    environment = tester.env_generator.generate_random_environment(
                                        min_obstacles=4, 
                                        max_obstacles=10
                                    )
                                    
                                    result = tester.run_single_test(
                                        environment=environment,
                                        algorithm_params=config,
                                        safety_margin=safety_margin,
                                        verbose=False
                                    )
                                    
                                    all_results.append(result)
                                    
                                except Exception as e:
                                    print(f"    Test {test_count} failed: {e}")
                                    continue
        
        df = pd.DataFrame([asdict(result) for result in all_results])
        analyze_extensive_results(df)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"mppi_medium_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
    else:
        print("Exiting...")
        
    print("\nTo analyze saved results later, load the CSV file:")
    print("df = pd.read_csv('mppi_results_TIMESTAMP.csv')")
    print("analyze_extensive_results(df)")