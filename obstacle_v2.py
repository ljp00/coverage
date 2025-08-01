"""
Advanced obstacle system for multi-agent coverage environment
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class BaseObstacle(ABC):
    """Abstract base class for obstacles"""

    def __init__(self, position: List[float], radius: float, obstacle_id: int = 0):
        self.initial_position = np.array(position, dtype=np.float32)
        self.position = np.array(position, dtype=np.float32)
        self.radius = radius
        self.obstacle_id = obstacle_id
        self.is_static = True

    @abstractmethod
    def update(self, dt: float, world_size: Tuple[float, float]):
        """Update obstacle position"""
        pass

    def get_distance_to_point(self, point: np.ndarray) -> float:
        """Calculate distance from obstacle surface to a point"""
        return max(0.0, np.linalg.norm(point - self.position) - self.radius)

    def is_colliding_with_point(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """Check if point is colliding with obstacle"""
        return np.linalg.norm(point - self.position) < (self.radius + safety_margin)

    def is_colliding_with_circle(self, center: np.ndarray, radius: float) -> bool:
        """Check if circle is colliding with obstacle"""
        return np.linalg.norm(center - self.position) < (self.radius + radius)


class StaticObstacle(BaseObstacle):
    """Static circular obstacle"""

    def __init__(self, position: List[float], radius: float, obstacle_id: int = 0):
        super().__init__(position, radius, obstacle_id)
        self.is_static = True

    def update(self, dt: float, world_size: Tuple[float, float]):
        """Static obstacles don't move"""
        pass


class DynamicObstacle(BaseObstacle):
    """Base class for dynamic obstacles"""

    def __init__(self, position: List[float], radius: float, obstacle_id: int = 0):
        super().__init__(position, radius, obstacle_id)
        self.is_static = False
        self.velocity = np.zeros(2, dtype=np.float32)
        self.time = 0.0

    def _bound_position(self, world_size: Tuple[float, float]):
        """Keep obstacle within world boundaries"""
        self.position[0] = np.clip(self.position[0], self.radius, world_size[0] - self.radius)
        self.position[1] = np.clip(self.position[1], self.radius, world_size[1] - self.radius)


class OscillatingObstacle(DynamicObstacle):
    """Obstacle that oscillates around a center point"""

    def __init__(self, position: List[float], radius: float,
                 amplitude: List[float], frequency: List[float],
                 phase: List[float] = None, obstacle_id: int = 0):
        super().__init__(position, radius, obstacle_id)
        self.center = np.array(position, dtype=np.float32)
        self.amplitude = np.array(amplitude, dtype=np.float32)
        self.frequency = np.array(frequency, dtype=np.float32)
        self.phase = np.array(phase if phase else [0.0, 0.0], dtype=np.float32)

    def update(self, dt: float, world_size: Tuple[float, float]):
        """Update oscillating position"""
        self.time += dt

        # Calculate oscillating offset
        offset = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)
        self.position = self.center + offset

        # Update velocity for prediction
        self.velocity = (2 * np.pi * self.frequency * self.amplitude *
                         np.cos(2 * np.pi * self.frequency * self.time + self.phase))

        self._bound_position(world_size)


class PatrollingObstacle(DynamicObstacle):
    """Obstacle that patrols between waypoints"""

    def __init__(self, position: List[float], radius: float,
                 waypoints: List[List[float]], speed: float = 1.0, obstacle_id: int = 0):
        super().__init__(position, radius, obstacle_id)
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.speed = speed
        self.current_target = 0
        self.direction = 1  # 1 for forward, -1 for backward

        if len(self.waypoints) < 2:
            raise ValueError("Patrolling obstacle needs at least 2 waypoints")

    def update(self, dt: float, world_size: Tuple[float, float]):
        """Update patrolling position"""
        if len(self.waypoints) < 2:
            return

        target_pos = self.waypoints[self.current_target]
        direction_vec = target_pos - self.position
        distance_to_target = np.linalg.norm(direction_vec)

        if distance_to_target < 0.1:  # Reached waypoint
            if self.direction == 1:
                self.current_target += 1
                if self.current_target >= len(self.waypoints):
                    self.current_target = len(self.waypoints) - 2
                    self.direction = -1
            else:
                self.current_target -= 1
                if self.current_target < 0:
                    self.current_target = 1
                    self.direction = 1
        else:
            # Move towards target
            move_distance = min(self.speed * dt, distance_to_target)
            if distance_to_target > 0:
                direction_unit = direction_vec / distance_to_target
                self.position += direction_unit * move_distance
                self.velocity = direction_unit * self.speed

        self._bound_position(world_size)


class RandomObstacle(DynamicObstacle):
    """Obstacle with random movement"""

    def __init__(self, position: List[float], radius: float,
                 speed: float = 0.5, direction_change_interval: float = 3.0, obstacle_id: int = 0):
        super().__init__(position, radius, obstacle_id)
        self.speed = speed
        self.direction_change_interval = direction_change_interval
        self.time_since_direction_change = 0.0
        self.target_direction = np.random.uniform(0, 2 * np.pi)

    def update(self, dt: float, world_size: Tuple[float, float]):
        """Update random movement"""
        self.time_since_direction_change += dt

        # Change direction periodically
        if self.time_since_direction_change >= self.direction_change_interval:
            self.target_direction = np.random.uniform(0, 2 * np.pi)
            self.time_since_direction_change = 0.0

        # Move in current direction
        self.velocity = self.speed * np.array([
            np.cos(self.target_direction),
            np.sin(self.target_direction)
        ])

        new_position = self.position + self.velocity * dt

        # Bounce off walls
        if new_position[0] <= self.radius or new_position[0] >= world_size[0] - self.radius:
            self.velocity[0] *= -1
            self.target_direction = np.pi - self.target_direction
        if new_position[1] <= self.radius or new_position[1] >= world_size[1] - self.radius:
            self.velocity[1] *= -1
            self.target_direction = -self.target_direction

        self.position += self.velocity * dt
        self._bound_position(world_size)


class ObstacleManager:
    """Manager for all obstacles in the environment"""

    def __init__(self):
        self.obstacles: List[BaseObstacle] = []
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.obstacle_counter = 0

    def add_obstacle(self, obstacle: BaseObstacle):
        """Add an obstacle to the manager"""
        obstacle.obstacle_id = self.obstacle_counter
        self.obstacle_counter += 1

        self.obstacles.append(obstacle)
        if obstacle.is_static:
            self.static_obstacles.append(obstacle)
        else:
            self.dynamic_obstacles.append(obstacle)

    def create_from_config(self, obstacle_config: Dict[str, Any]):
        """Create obstacles from configuration"""
        # Create static obstacles
        for obs_config in obstacle_config.get('static_obstacles', []):
            obstacle = StaticObstacle(
                position=obs_config['position'],
                radius=obs_config['radius']
            )
            self.add_obstacle(obstacle)

        # Create dynamic obstacles
        for obs_config in obstacle_config.get('dynamic_obstacles', []):
            obs_type = obs_config.get('type', 'oscillating')

            if obs_type == 'oscillating':
                obstacle = OscillatingObstacle(
                    position=obs_config['position'],
                    radius=obs_config['radius'],
                    amplitude=obs_config['amplitude'],
                    frequency=obs_config['frequency'],
                    phase=obs_config.get('phase', [0.0, 0.0])
                )
            elif obs_type == 'patrolling':
                obstacle = PatrollingObstacle(
                    position=obs_config['position'],
                    radius=obs_config['radius'],
                    waypoints=obs_config['waypoints'],
                    speed=obs_config.get('speed', 1.0)
                )
            elif obs_type == 'random':
                obstacle = RandomObstacle(
                    position=obs_config['position'],
                    radius=obs_config['radius'],
                    speed=obs_config.get('speed', 0.5),
                    direction_change_interval=obs_config.get('direction_change_interval', 3.0)
                )
            else:
                continue  # Skip unknown obstacle types

            self.add_obstacle(obstacle)

    def update_all(self, dt: float, world_size: Tuple[float, float]):
        """Update all dynamic obstacles"""
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt, world_size)

    def get_nearest_obstacle(self, position: np.ndarray) -> Tuple[BaseObstacle, float]:
        """Get nearest obstacle and distance to it"""
        if not self.obstacles:
            return None, float('inf')

        min_distance = float('inf')
        nearest_obstacle = None

        for obstacle in self.obstacles:
            distance = obstacle.get_distance_to_point(position)
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle

        return nearest_obstacle, min_distance

    def check_collision_with_point(self, position: np.ndarray, safety_margin: float = 0.0) -> bool:
        """Check if point collides with any obstacle"""
        for obstacle in self.obstacles:
            if obstacle.is_colliding_with_point(position, safety_margin):
                return True
        return False

    def check_collision_with_circle(self, center: np.ndarray, radius: float) -> bool:
        """Check if circle collides with any obstacle"""
        for obstacle in self.obstacles:
            if obstacle.is_colliding_with_circle(center, radius):
                return True
        return False

    def get_obstacles_in_range(self, position: np.ndarray, range_radius: float) -> List[BaseObstacle]:
        """Get all obstacles within range of a position"""
        obstacles_in_range = []
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle.position)
            if distance <= range_radius + obstacle.radius:
                obstacles_in_range.append(obstacle)
        return obstacles_in_range

    def reset(self):
        """Reset all obstacles to initial positions"""
        for obstacle in self.obstacles:
            obstacle.position = obstacle.initial_position.copy()
            if hasattr(obstacle, 'time'):
                obstacle.time = 0.0
            if hasattr(obstacle, 'current_target'):
                obstacle.current_target = 0
                obstacle.direction = 1

    def __len__(self):
        return len(self.obstacles)

    def __iter__(self):
        return iter(self.obstacles)