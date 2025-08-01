"""
Comprehensive Multi-Agent Coverage Environment V5
完整的多智能体覆盖环境 V5

This module implements a comprehensive multi-agent coverage path planning environment
with advanced features including:
- Full OpenAI Gymnasium compliance
- Sophisticated observation and action spaces  
- Efficient coverage calculation system
- Robust collision detection and safety constraints
- Multi-objective reward function design
- Advanced visualization and animation support
- Performance monitoring and metrics
- Complete type annotations and documentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.backends.backend_agg as agg
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import copy
import os
from datetime import datetime
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class CoverageEnvV5(gym.Env):
    """
    Comprehensive Multi-Agent Coverage Environment V5
    
    A complete implementation of multi-agent coverage path planning environment
    with advanced features for research and training.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config: Any, render_mode: Optional[str] = None):
        """
        Initialize the coverage environment.
        
        Args:
            config: Configuration object containing all environment parameters
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        # Store configuration
        self.config = config.config if hasattr(config, 'config') else config
        self.render_mode = render_mode
        
        # Environment parameters
        self.world_size = tuple(self.config['environment']['size'])
        self.grid_resolution = self.config['environment']['grid_resolution']
        self.max_steps = self.config['environment']['max_steps']
        self.time_limit = self.config['environment'].get('time_limit', 300.0)
        
        # Agent parameters
        self.n_agents = self.config['agents']['n_agents']
        self.sensor_range = self.config['agents']['sensor_range']
        self.min_distance = self.config['agents']['min_distance']
        self.max_velocity = self.config['agents']['max_velocity']
        self.agent_radius = self.config['agents']['agent_radius']
        
        # Coverage system parameters
        self.coverage_radius = self.sensor_range * 0.8  # Effective coverage radius
        self.grid_size_x = int(self.world_size[0] * self.grid_resolution / 10.0)
        self.grid_size_y = int(self.world_size[1] * self.grid_resolution / 10.0)
        
        # Laser sensor parameters for safety
        self.num_lasers = 16
        self.laser_range = 2.0
        
        # Reward parameters
        self.reward_config = self.config['rewards']
        
        # Safety parameters
        self.safety_config = self.config['safety']
        
        # Initialize spaces
        self._setup_spaces()
        
        # Initialize environment state
        self._init_state()
        
        # Initialize obstacles
        self._init_obstacles()
        
        # Initialize visualization
        self._init_visualization()
        
        # Performance tracking
        self._init_metrics()
        
        print(f"Coverage Environment V5 initialized:")
        print(f"  World size: {self.world_size}")
        print(f"  Grid resolution: {self.grid_size_x}x{self.grid_size_y}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Observation dim: {self.observation_space.shape[0]}")
        print(f"  Action dim: {self.action_space.shape[0]}")

    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        # Action space: continuous velocity control [vx, vy]
        self.action_space = spaces.Box(
            low=np.array([-self.max_velocity, -self.max_velocity]),
            high=np.array([self.max_velocity, self.max_velocity]),
            dtype=np.float32
        )
        
        # Observation space calculation
        obs_dim = (
            4 +                              # Self state: [x, y, vx, vy] normalized
            (self.n_agents - 1) * 2 +        # Other agents relative positions
            self.num_lasers +                # Laser sensor readings
            9 +                              # Local coverage info (3x3 grid)
            4 +                              # Boundary distances
            3 +                              # Global task state
            self.n_agents                    # Agent-specific coverage contributions
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _init_state(self) -> None:
        """Initialize environment state variables."""
        # Agent states
        self.agent_positions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_trajectories = [[] for _ in range(self.n_agents)]
        
        # Coverage system
        self.coverage_grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=bool)
        self.agent_coverage_grids = np.zeros((self.n_agents, self.grid_size_x, self.grid_size_y), dtype=bool)
        self.coverage_timestamps = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.float32)
        
        # Laser sensors
        self.laser_readings = np.full((self.n_agents, self.num_lasers), self.laser_range, dtype=np.float32)
        
        # Environment state
        self.step_count = 0
        self.episode_time = 0.0
        self.total_coverage_area = self.grid_size_x * self.grid_size_y
        
        # Collision and safety tracking
        self.collision_history = []
        self.safety_violations = []

    def _init_obstacles(self) -> None:
        """Initialize static and dynamic obstacles."""
        self.static_obstacles = []
        for obs_config in self.config['obstacles']['static_obstacles']:
            self.static_obstacles.append({
                'position': np.array(obs_config['position'], dtype=np.float32),
                'radius': obs_config['radius'],
                'is_dynamic': False
            })

        self.dynamic_obstacles = []
        for obs_config in self.config['obstacles']['dynamic_obstacles']:
            obstacle = {
                'initial_position': np.array(obs_config['position'], dtype=np.float32),
                'current_position': np.array(obs_config['position'], dtype=np.float32),
                'radius': obs_config['radius'],
                'type': obs_config['type'],
                'is_dynamic': True
            }

            if obs_config['type'] == 'oscillating':
                obstacle.update({
                    'amplitude': np.array(obs_config['amplitude'], dtype=np.float32),
                    'frequency': np.array(obs_config['frequency'], dtype=np.float32),
                    'phase': np.array(obs_config['phase'], dtype=np.float32)
                })

            self.dynamic_obstacles.append(obstacle)

        self.all_obstacles = self.static_obstacles + self.dynamic_obstacles

    def _init_visualization(self) -> None:
        """Initialize visualization components."""
        self.animation_frames = []
        self.frame_count = 0
        
        if self.render_mode is not None:
            # Create output directory for animations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"coverage_v5_simulation_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Color configuration
            self.colors = {
                'background': '#FFFFFF',
                'covered_area': '#90EE90',
                'uncovered_area': '#F0F0F0',
                'grid_lines': '#CCCCCC',
                'agents': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'],
                'trajectories': ['#800000', '#008000', '#000080', '#808000', '#800080', '#008080'],
                'sensor_range': '#00FF0040',
                'safety_zone': '#FFFF0040',
                'static_obstacle': '#404040',
                'dynamic_obstacle': '#8B4513',
                'laser_beam': '#FF000030'
            }

    def _init_metrics(self) -> None:
        """Initialize performance monitoring metrics."""
        self.metrics = {
            'coverage_history': [],
            'reward_history': [],
            'collision_count': 0,
            'safety_violations_count': 0,
            'efficiency_scores': [],
            'agent_contributions': [[] for _ in range(self.n_agents)],
            'exploration_entropy': [],
            'coverage_rate_per_step': [],
            'energy_consumption': [[] for _ in range(self.n_agents)]
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observations: List of initial observations for each agent
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset all state variables
        self._init_state()
        
        # Generate safe initial positions for agents
        self._generate_initial_positions()
        
        # Update initial sensor readings
        self._update_laser_sensors()
        
        # Initialize metrics
        self._init_metrics()
        
        # Get initial observations
        observations = self._get_observations()
        
        # Initial rendering
        if self.render_mode is not None:
            self.render()
        
        info = {
            'coverage_percentage': self.get_coverage_percentage(),
            'step_count': self.step_count,
            'agent_positions': self.agent_positions.copy(),
            'collision_status': [False] * self.n_agents
        }
        
        return observations, info

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Ensure actions are properly formatted
        actions = [np.array(action, dtype=np.float32) for action in actions]
        
        # Update dynamic obstacles
        self._update_dynamic_obstacles()
        
        # Apply actions and update agent states
        collision_status = self._update_agent_states(actions)
        
        # Update sensor systems
        self._update_laser_sensors()
        
        # Update coverage system
        newly_covered_cells = self._update_coverage_system()
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions, collision_status, newly_covered_cells)
        
        # Update metrics
        self._update_metrics(rewards, collision_status)
        
        # Check termination conditions
        self.step_count += 1
        self.episode_time += 1.0  # Assuming 1 time unit per step
        
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        # Get new observations
        observations = self._get_observations()
        
        # Render if needed
        if self.render_mode is not None:
            self.render()
        
        # Prepare info dictionary
        info = {
            'coverage_percentage': self.get_coverage_percentage(),
            'step_count': self.step_count,
            'collision_status': collision_status,
            'newly_covered_cells': newly_covered_cells,
            'agent_positions': self.agent_positions.copy(),
            'safety_violations': len(self.safety_violations),
            'efficiency_score': self._calculate_efficiency_score()
        }
        
        return observations, rewards, terminated, truncated, info

    def _generate_initial_positions(self) -> None:
        """Generate safe initial positions for all agents."""
        for i in range(self.n_agents):
            max_attempts = 100
            for attempt in range(max_attempts):
                # Generate random position with margins
                position = np.array([
                    np.random.uniform(self.agent_radius + 0.5, 
                                    self.world_size[0] - self.agent_radius - 0.5),
                    np.random.uniform(self.agent_radius + 0.5, 
                                    self.world_size[1] - self.agent_radius - 0.5)
                ], dtype=np.float32)
                
                if self._is_position_safe(position, exclude_agent=i):
                    self.agent_positions[i] = position
                    self.agent_trajectories[i] = [position.copy()]
                    break
            else:
                # Fallback position if no safe position found
                self.agent_positions[i] = np.array([
                    1.0 + i * 0.5, 1.0 + i * 0.3
                ], dtype=np.float32)
                self.agent_trajectories[i] = [self.agent_positions[i].copy()]

    def _is_position_safe(self, position: np.ndarray, exclude_agent: int = -1) -> bool:
        """Check if a position is safe from collisions."""
        # Check against other agents
        for j, other_pos in enumerate(self.agent_positions):
            if j != exclude_agent and j < self.n_agents:
                if np.linalg.norm(position - other_pos) < self.min_distance:
                    return False
        
        # Check against obstacles
        for obstacle in self.all_obstacles:
            obs_pos = obstacle['current_position'] if obstacle['is_dynamic'] else obstacle['position']
            distance = np.linalg.norm(position - obs_pos)
            if distance < (obstacle['radius'] + self.agent_radius + 0.2):
                return False
        
        return True

    def _update_dynamic_obstacles(self) -> None:
        """Update positions of dynamic obstacles."""
        for obstacle in self.dynamic_obstacles:
            if obstacle['type'] == 'oscillating':
                time_factor = self.episode_time * 0.1
                displacement = obstacle['amplitude'] * np.sin(
                    obstacle['frequency'] * time_factor + obstacle['phase']
                )
                obstacle['current_position'] = obstacle['initial_position'] + displacement
                
                # Enforce boundary constraints
                obstacle['current_position'] = np.clip(
                    obstacle['current_position'],
                    [obstacle['radius'], obstacle['radius']],
                    [self.world_size[0] - obstacle['radius'], 
                     self.world_size[1] - obstacle['radius']]
                )

    def _update_agent_states(self, actions: List[np.ndarray]) -> List[bool]:
        """
        Update agent positions and velocities based on actions.
        
        Args:
            actions: List of velocity commands for each agent
            
        Returns:
            collision_status: List indicating collision status for each agent
        """
        collision_status = [False] * self.n_agents
        
        for i, action in enumerate(actions):
            # Clip action to valid range
            action = np.clip(action, -self.max_velocity, self.max_velocity)
            
            # Update velocity
            self.agent_velocities[i] = action
            
            # Calculate new position
            new_position = self.agent_positions[i] + action
            
            # Apply boundary constraints
            new_position = self._enforce_boundary_constraints(new_position)
            
            # Check for collisions and apply safety constraints
            safe_position, collision = self._enforce_safety_constraints(
                new_position, i, self.agent_positions[i]
            )
            
            # Update position
            self.agent_positions[i] = safe_position
            collision_status[i] = collision
            
            # Update trajectory
            self.agent_trajectories[i].append(safe_position.copy())
            
            # Limit trajectory length for memory efficiency
            if len(self.agent_trajectories[i]) > 1000:
                self.agent_trajectories[i] = self.agent_trajectories[i][-1000:]
        
        return collision_status

    def _enforce_boundary_constraints(self, position: np.ndarray) -> np.ndarray:
        """Enforce world boundary constraints."""
        return np.clip(
            position,
            [self.agent_radius, self.agent_radius],
            [self.world_size[0] - self.agent_radius, 
             self.world_size[1] - self.agent_radius]
        )

    def _enforce_safety_constraints(self, new_position: np.ndarray, agent_idx: int, 
                                  current_position: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Enforce safety constraints and detect collisions.
        
        Args:
            new_position: Proposed new position
            agent_idx: Index of the current agent
            current_position: Current position of the agent
            
        Returns:
            safe_position: Adjusted safe position
            collision_detected: Whether a collision was detected
        """
        collision_detected = False
        safe_position = new_position.copy()
        
        # Check collision with other agents
        for j, other_pos in enumerate(self.agent_positions):
            if j != agent_idx:
                distance = np.linalg.norm(safe_position - other_pos)
                if distance < self.min_distance:
                    collision_detected = True
                    # Push away from collision
                    direction = safe_position - other_pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        safe_position = other_pos + direction * self.min_distance
                    else:
                        # Random direction if positions are identical
                        angle = np.random.uniform(0, 2 * np.pi)
                        direction = np.array([np.cos(angle), np.sin(angle)])
                        safe_position = other_pos + direction * self.min_distance
        
        # Check collision with obstacles
        for obstacle in self.all_obstacles:
            obs_pos = obstacle['current_position'] if obstacle['is_dynamic'] else obstacle['position']
            distance = np.linalg.norm(safe_position - obs_pos)
            min_distance = obstacle['radius'] + self.agent_radius + 0.1
            
            if distance < min_distance:
                collision_detected = True
                # Push away from obstacle
                direction = safe_position - obs_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    safe_position = obs_pos + direction * min_distance
                else:
                    # If exactly on obstacle center, move to current position
                    safe_position = current_position
        
        # Ensure final position is within boundaries
        safe_position = self._enforce_boundary_constraints(safe_position)
        
        return safe_position, collision_detected

    def _update_laser_sensors(self) -> None:
        """Update laser sensor readings for all agents."""
        for i in range(self.n_agents):
            self.laser_readings[i] = self._compute_laser_readings(i)

    def _compute_laser_readings(self, agent_idx: int) -> np.ndarray:
        """
        Compute laser sensor readings for a specific agent.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            laser_readings: Array of distances for each laser beam
        """
        agent_pos = self.agent_positions[agent_idx]
        readings = np.full(self.num_lasers, self.laser_range, dtype=np.float32)
        
        # Compute laser angles
        angles = np.linspace(0, 2 * np.pi, self.num_lasers, endpoint=False)
        
        for laser_idx, angle in enumerate(angles):
            ray_direction = np.array([np.cos(angle), np.sin(angle)])
            min_distance = self.laser_range
            
            # Check intersection with other agents
            for j, other_pos in enumerate(self.agent_positions):
                if j != agent_idx:
                    intersection_dist = self._ray_circle_intersection(
                        agent_pos, ray_direction, other_pos, self.agent_radius
                    )
                    if intersection_dist is not None and intersection_dist < min_distance:
                        min_distance = intersection_dist
            
            # Check intersection with obstacles
            for obstacle in self.all_obstacles:
                obs_pos = obstacle['current_position'] if obstacle['is_dynamic'] else obstacle['position']
                intersection_dist = self._ray_circle_intersection(
                    agent_pos, ray_direction, obs_pos, obstacle['radius']
                )
                if intersection_dist is not None and intersection_dist < min_distance:
                    min_distance = intersection_dist
            
            # Check intersection with boundaries
            boundary_dist = self._ray_boundary_intersection(agent_pos, ray_direction)
            if boundary_dist < min_distance:
                min_distance = boundary_dist
            
            readings[laser_idx] = min_distance
        
        return readings

    def _ray_circle_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                               circle_center: np.ndarray, circle_radius: float) -> Optional[float]:
        """Compute intersection distance between ray and circle."""
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - circle_radius * circle_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        if t1 > 0.01:  # Small epsilon to avoid self-intersection
            return t1
        elif t2 > 0.01:
            return t2
        else:
            return None

    def _ray_boundary_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> float:
        """Compute intersection distance between ray and world boundaries."""
        distances = []
        
        # Check intersection with four boundaries
        if abs(ray_dir[0]) > 1e-8:
            # Left boundary (x = 0)
            t = -ray_origin[0] / ray_dir[0]
            if t > 0:
                y = ray_origin[1] + t * ray_dir[1]
                if 0 <= y <= self.world_size[1]:
                    distances.append(t)
            
            # Right boundary (x = world_size[0])
            t = (self.world_size[0] - ray_origin[0]) / ray_dir[0]
            if t > 0:
                y = ray_origin[1] + t * ray_dir[1]
                if 0 <= y <= self.world_size[1]:
                    distances.append(t)
        
        if abs(ray_dir[1]) > 1e-8:
            # Bottom boundary (y = 0)
            t = -ray_origin[1] / ray_dir[1]
            if t > 0:
                x = ray_origin[0] + t * ray_dir[0]
                if 0 <= x <= self.world_size[0]:
                    distances.append(t)
            
            # Top boundary (y = world_size[1])
            t = (self.world_size[1] - ray_origin[1]) / ray_dir[1]
            if t > 0:
                x = ray_origin[0] + t * ray_dir[0]
                if 0 <= x <= self.world_size[0]:
                    distances.append(t)
        
        return min(distances) if distances else self.laser_range

    def _update_coverage_system(self) -> int:
        """
        Update coverage grid based on agent positions.
        
        Returns:
            newly_covered_cells: Number of newly covered cells
        """
        newly_covered_cells = 0
        
        for i, agent_pos in enumerate(self.agent_positions):
            # Calculate grid indices for cells within coverage radius
            min_gx = max(0, int((agent_pos[0] - self.coverage_radius) * self.grid_size_x / self.world_size[0]))
            max_gx = min(self.grid_size_x, int((agent_pos[0] + self.coverage_radius) * self.grid_size_x / self.world_size[0]) + 1)
            min_gy = max(0, int((agent_pos[1] - self.coverage_radius) * self.grid_size_y / self.world_size[1]))
            max_gy = min(self.grid_size_y, int((agent_pos[1] + self.coverage_radius) * self.grid_size_y / self.world_size[1]) + 1)
            
            for gx in range(min_gx, max_gx):
                for gy in range(min_gy, max_gy):
                    # Calculate world position of grid cell center
                    cell_world_x = (gx + 0.5) * self.world_size[0] / self.grid_size_x
                    cell_world_y = (gy + 0.5) * self.world_size[1] / self.grid_size_y
                    cell_pos = np.array([cell_world_x, cell_world_y])
                    
                    # Check if cell is within coverage radius
                    distance = np.linalg.norm(agent_pos - cell_pos)
                    if distance <= self.coverage_radius:
                        # Mark cell as covered by this agent
                        if not self.agent_coverage_grids[i, gx, gy]:
                            self.agent_coverage_grids[i, gx, gy] = True
                        
                        # Mark cell as globally covered
                        if not self.coverage_grid[gx, gy]:
                            self.coverage_grid[gx, gy] = True
                            self.coverage_timestamps[gx, gy] = self.episode_time
                            newly_covered_cells += 1
        
        return newly_covered_cells

    def _calculate_rewards(self, actions: List[np.ndarray], collision_status: List[bool], 
                         newly_covered_cells: int) -> List[float]:
        """
        Calculate comprehensive rewards for each agent.
        
        Args:
            actions: Actions taken by each agent
            collision_status: Collision status for each agent
            newly_covered_cells: Number of newly covered cells
            
        Returns:
            rewards: List of rewards for each agent
        """
        rewards = []
        
        for i in range(self.n_agents):
            reward = 0.0
            
            # 1. Coverage reward
            coverage_reward = self._calculate_coverage_reward(i, newly_covered_cells)
            reward += coverage_reward
            
            # 2. Exploration reward
            exploration_reward = self._calculate_exploration_reward(i)
            reward += exploration_reward
            
            # 3. Collision penalty
            if collision_status[i]:
                reward += self.reward_config['collision_penalty']
            
            # 4. Safety reward/penalty
            safety_reward = self._calculate_safety_reward(i)
            reward += safety_reward
            
            # 5. Efficiency reward
            efficiency_reward = self._calculate_efficiency_reward(i, actions[i])
            reward += efficiency_reward
            
            # 6. Cooperation reward
            cooperation_reward = self._calculate_cooperation_reward(i)
            reward += cooperation_reward
            
            # 7. Time penalty
            reward += self.reward_config['time_penalty']
            
            rewards.append(reward)
        
        return rewards

    def _calculate_coverage_reward(self, agent_idx: int, newly_covered_cells: int) -> float:
        """Calculate coverage-based reward for an agent."""
        # Base coverage reward
        agent_coverage = np.sum(self.agent_coverage_grids[agent_idx])
        coverage_reward = agent_coverage * self.reward_config['new_coverage'] / 1000.0
        
        # New coverage bonus
        if newly_covered_cells > 0:
            coverage_reward += newly_covered_cells * self.reward_config['new_coverage']
        
        return coverage_reward

    def _calculate_exploration_reward(self, agent_idx: int) -> float:
        """Calculate exploration reward to encourage visiting new areas."""
        agent_pos = self.agent_positions[agent_idx]
        
        # Calculate local coverage density (lower is better for exploration)
        local_coverage = self._compute_local_coverage_density(agent_pos)
        exploration_reward = (1.0 - local_coverage) * self.reward_config['efficiency_bonus']
        
        return exploration_reward

    def _calculate_safety_reward(self, agent_idx: int) -> float:
        """Calculate safety-based reward/penalty."""
        min_laser_distance = np.min(self.laser_readings[agent_idx])
        safety_threshold = self.min_distance * self.safety_config['warning_distance_multiplier']
        
        if min_laser_distance < safety_threshold:
            # Penalty for being too close to obstacles/agents
            penalty_factor = (safety_threshold - min_laser_distance) / safety_threshold
            return self.reward_config['safety_penalty_base'] * penalty_factor
        else:
            # Small reward for maintaining safe distance
            return 0.1
        
    def _calculate_efficiency_reward(self, agent_idx: int, action: np.ndarray) -> float:
        """Calculate efficiency reward based on energy consumption and progress."""
        # Energy penalty based on action magnitude
        energy_penalty = -np.linalg.norm(action) * self.reward_config['energy_penalty_factor']
        
        # Efficiency bonus for balanced speed and coverage
        speed = np.linalg.norm(self.agent_velocities[agent_idx])
        local_coverage = self._compute_local_coverage_density(self.agent_positions[agent_idx])
        
        # Reward high speed in unexplored areas, lower speed in explored areas
        efficiency_bonus = speed * (1.0 - local_coverage) * self.reward_config['efficiency_bonus']
        
        return energy_penalty + efficiency_bonus

    def _calculate_cooperation_reward(self, agent_idx: int) -> float:
        """Calculate cooperation reward to encourage coordination."""
        cooperation_reward = 0.0
        
        # Check for overlap with other agents' coverage
        for j in range(self.n_agents):
            if j != agent_idx:
                distance = np.linalg.norm(self.agent_positions[agent_idx] - self.agent_positions[j])
                if distance < 2 * self.coverage_radius:
                    # Penalty for overlapping coverage areas
                    overlap_factor = max(0, (2 * self.coverage_radius - distance) / (2 * self.coverage_radius))
                    cooperation_reward += self.reward_config['overlap_penalty_base'] * overlap_factor
        
        return cooperation_reward

    def _compute_local_coverage_density(self, position: np.ndarray, radius: float = 1.0) -> float:
        """Compute local coverage density around a position."""
        # Sample points in a circle around the position
        sample_points = []
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for r in np.linspace(0.1, radius, 4):
                sample_x = position[0] + r * np.cos(angle)
                sample_y = position[1] + r * np.sin(angle)
                if (0 <= sample_x <= self.world_size[0] and 
                    0 <= sample_y <= self.world_size[1]):
                    sample_points.append([sample_x, sample_y])
        
        if not sample_points:
            return 0.0
        
        # Check coverage status of sample points
        covered_count = 0
        for point in sample_points:
            gx = int(point[0] * self.grid_size_x / self.world_size[0])
            gy = int(point[1] * self.grid_size_y / self.world_size[1])
            gx = np.clip(gx, 0, self.grid_size_x - 1)
            gy = np.clip(gy, 0, self.grid_size_y - 1)
            
            if self.coverage_grid[gx, gy]:
                covered_count += 1
        
        return covered_count / len(sample_points)

    def _get_observations(self) -> List[np.ndarray]:
        """
        Generate observations for all agents.
        
        Returns:
            observations: List of observation arrays for each agent
        """
        observations = []
        
        for i in range(self.n_agents):
            obs = []
            
            # 1. Self state (normalized)
            pos = self.agent_positions[i]
            vel = self.agent_velocities[i]
            obs.extend([
                pos[0] / self.world_size[0],
                pos[1] / self.world_size[1],
                vel[0] / self.max_velocity,
                vel[1] / self.max_velocity
            ])
            
            # 2. Other agents relative positions (normalized)
            for j in range(self.n_agents):
                if j != i:
                    rel_pos = self.agent_positions[j] - pos
                    obs.extend([
                        rel_pos[0] / self.world_size[0],
                        rel_pos[1] / self.world_size[1]
                    ])
            
            # 3. Laser sensor readings (normalized)
            obs.extend(self.laser_readings[i] / self.laser_range)
            
            # 4. Local coverage information (3x3 grid around agent)
            local_coverage = self._get_local_coverage_grid(pos)
            obs.extend(local_coverage.flatten())
            
            # 5. Boundary distances (normalized)
            boundary_distances = [
                pos[0] / self.world_size[0],                    # Distance to left
                (self.world_size[0] - pos[0]) / self.world_size[0],  # Distance to right
                pos[1] / self.world_size[1],                    # Distance to bottom
                (self.world_size[1] - pos[1]) / self.world_size[1]   # Distance to top
            ]
            obs.extend(boundary_distances)
            
            # 6. Global task state
            global_coverage = self.get_coverage_percentage() / 100.0
            step_progress = self.step_count / self.max_steps
            time_progress = self.episode_time / self.time_limit
            obs.extend([global_coverage, step_progress, time_progress])
            
            # 7. Agent-specific coverage contributions
            agent_contributions = []
            for j in range(self.n_agents):
                contribution = np.sum(self.agent_coverage_grids[j]) / self.total_coverage_area
                agent_contributions.append(contribution)
            obs.extend(agent_contributions)
            
            # Ensure observation matches expected dimension
            expected_dim = self.observation_space.shape[0]
            if len(obs) != expected_dim:
                if len(obs) < expected_dim:
                    obs.extend([0.0] * (expected_dim - len(obs)))
                else:
                    obs = obs[:expected_dim]
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations

    def _get_local_coverage_grid(self, position: np.ndarray, grid_size: int = 3) -> np.ndarray:
        """Get local coverage grid around agent position."""
        local_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Calculate cell size for local grid
        cell_size = self.coverage_radius * 2 / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate world position of local grid cell
                local_x = position[0] + (i - grid_size//2) * cell_size
                local_y = position[1] + (j - grid_size//2) * cell_size
                
                # Check if position is within world boundaries
                if (0 <= local_x <= self.world_size[0] and 
                    0 <= local_y <= self.world_size[1]):
                    # Convert to global grid coordinates
                    gx = int(local_x * self.grid_size_x / self.world_size[0])
                    gy = int(local_y * self.grid_size_y / self.world_size[1])
                    gx = np.clip(gx, 0, self.grid_size_x - 1)
                    gy = np.clip(gy, 0, self.grid_size_y - 1)
                    
                    local_grid[i, j] = float(self.coverage_grid[gx, gy])
        
        return local_grid

    def _update_metrics(self, rewards: List[float], collision_status: List[bool]) -> None:
        """Update performance metrics."""
        # Coverage metrics
        coverage_percentage = self.get_coverage_percentage()
        self.metrics['coverage_history'].append(coverage_percentage)
        
        # Reward metrics
        self.metrics['reward_history'].append(rewards.copy())
        
        # Collision metrics
        self.metrics['collision_count'] += sum(collision_status)
        
        # Agent contribution metrics
        for i in range(self.n_agents):
            agent_coverage = np.sum(self.agent_coverage_grids[i]) / self.total_coverage_area * 100
            self.metrics['agent_contributions'][i].append(agent_coverage)
        
        # Efficiency metrics
        efficiency_score = self._calculate_efficiency_score()
        self.metrics['efficiency_scores'].append(efficiency_score)
        
        # Energy consumption
        for i in range(self.n_agents):
            energy = np.linalg.norm(self.agent_velocities[i])
            self.metrics['energy_consumption'][i].append(energy)

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        if self.step_count == 0:
            return 0.0
        
        coverage_rate = self.get_coverage_percentage()
        time_efficiency = coverage_rate / max(self.step_count, 1)
        
        # Factor in energy efficiency
        total_energy = sum(np.sum(self.metrics['energy_consumption'][i]) 
                          for i in range(self.n_agents))
        energy_efficiency = coverage_rate / max(total_energy, 0.001)
        
        return (time_efficiency + energy_efficiency) / 2

    def _check_termination(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if full coverage achieved
        if self.get_coverage_percentage() >= 99.9:
            return True
        
        # Terminate if time limit exceeded
        if self.episode_time >= self.time_limit:
            return True
        
        return False

    def get_coverage_percentage(self) -> float:
        """Get current coverage percentage."""
        covered_cells = np.sum(self.coverage_grid)
        return (covered_cells / self.total_coverage_area) * 100.0

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            rgb_array: RGB array if render_mode is 'rgb_array', else None
        """
        if self.render_mode is None:
            return None
        
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, self.world_size[0])
            ax.set_ylim(0, self.world_size[1])
            ax.set_aspect('equal')
            
            # Draw coverage grid
            self._draw_coverage_grid(ax)
            
            # Draw obstacles
            self._draw_obstacles(ax)
            
            # Draw agents
            self._draw_agents(ax)
            
            # Add information panel
            self._add_info_panel(ax)
            
            # Set title and labels
            coverage_rate = self.get_coverage_percentage()
            ax.set_title(f'Multi-Agent Coverage Environment V5\n'
                        f'Step: {self.step_count}, Coverage: {coverage_rate:.1f}%',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Convert to RGB array
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf)
            
            # Save frame if in rgb_array mode
            if self.render_mode == 'rgb_array':
                # Save PNG frame
                frame_filename = f"frame_{self.frame_count:04d}.png"
                frame_path = os.path.join(self.output_dir, frame_filename)
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                
                # Save to animation frames
                buf_img = io.BytesIO()
                plt.savefig(buf_img, format='png', dpi=100, bbox_inches='tight')
                buf_img.seek(0)
                img = Image.open(buf_img)
                self.animation_frames.append(img.copy())
                buf_img.close()
                
                self.frame_count += 1
            
            plt.close(fig)
            return image_array[:, :, :3]  # Remove alpha channel
            
        except Exception as e:
            print(f"Rendering error: {e}")
            return np.zeros((800, 600, 3), dtype=np.uint8)

    def _draw_coverage_grid(self, ax) -> None:
        """Draw the coverage grid."""
        cell_width = self.world_size[0] / self.grid_size_x
        cell_height = self.world_size[1] / self.grid_size_y
        
        for gx in range(self.grid_size_x):
            for gy in range(self.grid_size_y):
                x = gx * cell_width
                y = gy * cell_height
                
                color = (self.colors['covered_area'] if self.coverage_grid[gx, gy] 
                        else self.colors['uncovered_area'])
                
                rect = patches.Rectangle(
                    (x, y), cell_width, cell_height,
                    facecolor=color, edgecolor=self.colors['grid_lines'],
                    linewidth=0.1, alpha=0.7
                )
                ax.add_patch(rect)

    def _draw_obstacles(self, ax) -> None:
        """Draw static and dynamic obstacles."""
        # Static obstacles
        for obstacle in self.static_obstacles:
            circle = patches.Circle(
                obstacle['position'], obstacle['radius'],
                facecolor=self.colors['static_obstacle'],
                edgecolor='black', linewidth=2, alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(obstacle['position'][0], obstacle['position'][1], 'S',
                   ha='center', va='center', fontsize=10, 
                   fontweight='bold', color='white')
        
        # Dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            circle = patches.Circle(
                obstacle['current_position'], obstacle['radius'],
                facecolor=self.colors['dynamic_obstacle'],
                edgecolor='black', linewidth=2, alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(obstacle['current_position'][0], obstacle['current_position'][1], 'D',
                   ha='center', va='center', fontsize=10, 
                   fontweight='bold', color='white')

    def _draw_agents(self, ax) -> None:
        """Draw agents with their sensors, trajectories, and status."""
        for i in range(self.n_agents):
            pos = self.agent_positions[i]
            vel = self.agent_velocities[i]
            color = self.colors['agents'][i % len(self.colors['agents'])]
            
            # Draw trajectory
            if len(self.agent_trajectories[i]) > 1:
                trajectory = np.array(self.agent_trajectories[i])
                ax.plot(trajectory[-100:, 0], trajectory[-100:, 1],  # Last 100 points
                       color=self.colors['trajectories'][i % len(self.colors['trajectories'])],
                       alpha=0.6, linewidth=2)
            
            # Draw coverage range
            coverage_circle = patches.Circle(
                pos, self.coverage_radius,
                facecolor=self.colors['sensor_range'],
                edgecolor=color, linewidth=1.5, alpha=0.2
            )
            ax.add_patch(coverage_circle)
            
            # Draw safety zone
            safety_circle = patches.Circle(
                pos, self.min_distance,
                facecolor=self.colors['safety_zone'],
                edgecolor='orange', linewidth=1, alpha=0.2
            )
            ax.add_patch(safety_circle)
            
            # Draw agent body
            agent_circle = patches.Circle(
                pos, self.agent_radius,
                facecolor=color, edgecolor='black',
                linewidth=2, alpha=0.9
            )
            ax.add_patch(agent_circle)
            
            # Draw velocity vector
            if np.linalg.norm(vel) > 0.01:
                ax.arrow(pos[0], pos[1], vel[0] * 0.5, vel[1] * 0.5,
                        head_width=0.05, head_length=0.05, 
                        fc=color, ec=color, alpha=0.8)
            
            # Add agent ID
            ax.text(pos[0], pos[1], str(i), ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

    def _add_info_panel(self, ax) -> None:
        """Add information panel to the plot."""
        info_text = []
        
        # Coverage information
        coverage_rate = self.get_coverage_percentage()
        info_text.append(f"Coverage: {coverage_rate:.1f}%")
        
        # Progress information
        info_text.append(f"Step: {self.step_count}/{self.max_steps}")
        info_text.append(f"Time: {self.episode_time:.1f}/{self.time_limit:.1f}")
        
        # Agent status
        for i in range(self.n_agents):
            vel_mag = np.linalg.norm(self.agent_velocities[i])
            agent_coverage = np.sum(self.agent_coverage_grids[i]) / self.total_coverage_area * 100
            info_text.append(f"Agent {i}: v={vel_mag:.2f}, cov={agent_coverage:.1f}%")
        
        # Performance metrics
        if self.metrics['efficiency_scores']:
            efficiency = self.metrics['efficiency_scores'][-1]
            info_text.append(f"Efficiency: {efficiency:.2f}")
        
        info_text.append(f"Collisions: {self.metrics['collision_count']}")
        
        # Display information
        info_str = '\n'.join(info_text)
        ax.text(0.98, 0.98, info_str, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, fontfamily='monospace')

    def close(self) -> None:
        """Close the environment and save results."""
        if self.render_mode is not None and self.animation_frames:
            self._save_animation()
            self._save_metrics()
        plt.close('all')

    def _save_animation(self) -> None:
        """Save animation as GIF."""
        if not self.animation_frames:
            return
        
        try:
            gif_filename = "coverage_v5_simulation.gif"
            gif_path = os.path.join(self.output_dir, gif_filename)
            
            self.animation_frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.animation_frames[1:],
                duration=200,
                loop=0
            )
            
            print(f"Animation saved: {gif_path}")
            print(f"Frames saved: {self.frame_count}")
            
        except Exception as e:
            print(f"Error saving animation: {e}")

    def _save_metrics(self) -> None:
        """Save performance metrics plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Coverage history
            axes[0, 0].plot(self.metrics['coverage_history'], 'b-', linewidth=2)
            axes[0, 0].set_title('Coverage Rate History')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Coverage Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Reward history
            if self.metrics['reward_history']:
                rewards_array = np.array(self.metrics['reward_history'])
                for i in range(self.n_agents):
                    axes[0, 1].plot(rewards_array[:, i], label=f'Agent {i}', linewidth=2)
                axes[0, 1].set_title('Reward History')
                axes[0, 1].set_xlabel('Steps')
                axes[0, 1].set_ylabel('Reward')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Efficiency scores
            if self.metrics['efficiency_scores']:
                axes[1, 0].plot(self.metrics['efficiency_scores'], 'g-', linewidth=2)
                axes[1, 0].set_title('Efficiency Score')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Efficiency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Agent contributions
            for i in range(self.n_agents):
                if self.metrics['agent_contributions'][i]:
                    axes[1, 1].plot(self.metrics['agent_contributions'][i], 
                                   label=f'Agent {i}', linewidth=2)
            axes[1, 1].set_title('Agent Coverage Contributions')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Coverage Contribution (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            metrics_path = os.path.join(self.output_dir, 'metrics_v5.png')
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Metrics saved: {metrics_path}")
            
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def get_env_info(self) -> Dict:
        """Get comprehensive environment information."""
        return {
            'environment_version': 'CoverageEnvV5',
            'world_size': self.world_size,
            'grid_resolution': [self.grid_size_x, self.grid_size_y],
            'n_agents': self.n_agents,
            'observation_dim': self.observation_space.shape[0],
            'action_dim': self.action_space.shape[0],
            'max_steps': self.max_steps,
            'current_step': self.step_count,
            'coverage_percentage': self.get_coverage_percentage(),
            'total_collisions': self.metrics['collision_count'],
            'efficiency_score': (self.metrics['efficiency_scores'][-1] 
                               if self.metrics['efficiency_scores'] else 0.0)
        }