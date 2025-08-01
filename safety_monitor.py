"""
Safety monitoring system for multi-agent coverage environment
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from obstacle_v2 import ObstacleManager, BaseObstacle


class SafetyLevel:
    """Safety level enumeration"""
    SAFE = 0
    WARNING = 1
    DANGER = 2
    COLLISION = 3


class SafetyEvent:
    """Safety event data structure"""

    def __init__(self, agent_id: int, event_type: str, severity: int,
                 position: np.ndarray, details: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.event_type = event_type  # 'agent_collision', 'obstacle_collision', 'boundary_violation'
        self.severity = severity
        self.position = position.copy()
        self.details = details or {}
        self.timestamp = 0


class SafetyMonitor:
    """Monitor and assess safety violations in the environment"""

    def __init__(self, config: Dict[str, Any]):
        self.min_distance = config['agents']['min_distance']
        self.agent_radius = config['agents']['agent_radius']
        self.world_size = config['environment']['size']

        # Safety thresholds
        self.warning_distance = self.min_distance * config['safety']['warning_distance_multiplier']
        self.danger_distance = self.min_distance * config['safety']['danger_distance_multiplier']
        self.collision_threshold = self.min_distance * config['safety']['collision_threshold']

        # Event tracking
        self.safety_events: List[SafetyEvent] = []
        self.current_violations: Dict[int, List[SafetyEvent]] = {}
        self.statistics = {
            'total_collisions': 0,
            'agent_collisions': 0,
            'obstacle_collisions': 0,
            'boundary_violations': 0,
            'warning_events': 0
        }

    def assess_agent_safety(self, agent_positions: List[np.ndarray],
                            obstacle_manager: ObstacleManager) -> Dict[int, Dict[str, Any]]:
        """Assess safety for all agents"""
        safety_assessment = {}

        for i, position in enumerate(agent_positions):
            assessment = {
                'safety_level': SafetyLevel.SAFE,
                'violations': [],
                'distances': {
                    'nearest_agent': float('inf'),
                    'nearest_obstacle': float('inf'),
                    'boundary': self._get_boundary_distance(position)
                },
                'warnings': []
            }

            # Check agent-to-agent safety
            for j, other_position in enumerate(agent_positions):
                if i != j:
                    distance = np.linalg.norm(position - other_position)
                    assessment['distances']['nearest_agent'] = min(
                        assessment['distances']['nearest_agent'], distance
                    )

                    if distance < self.collision_threshold:
                        event = SafetyEvent(
                            agent_id=i,
                            event_type='agent_collision',
                            severity=SafetyLevel.COLLISION,
                            position=position,
                            details={'other_agent': j, 'distance': distance}
                        )
                        assessment['violations'].append(event)
                        assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.COLLISION)
                    elif distance < self.danger_distance:
                        assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.DANGER)
                        assessment['warnings'].append(f"Danger: Agent {j} at distance {distance:.2f}")
                    elif distance < self.warning_distance:
                        assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.WARNING)
                        assessment['warnings'].append(f"Warning: Agent {j} at distance {distance:.2f}")

            # Check agent-to-obstacle safety
            nearest_obstacle, obstacle_distance = obstacle_manager.get_nearest_obstacle(position)
            if nearest_obstacle:
                assessment['distances']['nearest_obstacle'] = obstacle_distance

                if obstacle_distance < 0.01:  # Collision
                    event = SafetyEvent(
                        agent_id=i,
                        event_type='obstacle_collision',
                        severity=SafetyLevel.COLLISION,
                        position=position,
                        details={'obstacle_id': nearest_obstacle.obstacle_id, 'distance': obstacle_distance}
                    )
                    assessment['violations'].append(event)
                    assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.COLLISION)
                elif obstacle_distance < self.danger_distance - nearest_obstacle.radius:
                    assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.DANGER)
                    assessment['warnings'].append(f"Danger: Obstacle at distance {obstacle_distance:.2f}")
                elif obstacle_distance < self.warning_distance - nearest_obstacle.radius:
                    assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.WARNING)
                    assessment['warnings'].append(f"Warning: Obstacle at distance {obstacle_distance:.2f}")

            # Check boundary violations
            boundary_distance = assessment['distances']['boundary']
            if boundary_distance < self.agent_radius:
                event = SafetyEvent(
                    agent_id=i,
                    event_type='boundary_violation',
                    severity=SafetyLevel.COLLISION,
                    position=position,
                    details={'boundary_distance': boundary_distance}
                )
                assessment['violations'].append(event)
                assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.COLLISION)
            elif boundary_distance < self.warning_distance:
                assessment['safety_level'] = max(assessment['safety_level'], SafetyLevel.WARNING)
                assessment['warnings'].append(f"Warning: Near boundary, distance {boundary_distance:.2f}")

            safety_assessment[i] = assessment

        # Update statistics and events
        self._update_statistics(safety_assessment)

        return safety_assessment

    def _get_boundary_distance(self, position: np.ndarray) -> float:
        """Calculate minimum distance to environment boundaries"""
        distances = [
            position[0],  # Left boundary
            self.world_size[0] - position[0],  # Right boundary
            position[1],  # Bottom boundary
            self.world_size[1] - position[1]  # Top boundary
        ]
        return min(distances)

    def _update_statistics(self, safety_assessment: Dict[int, Dict[str, Any]]):
        """Update safety statistics"""
        for agent_id, assessment in safety_assessment.items():
            for violation in assessment['violations']:
                self.safety_events.append(violation)

                if violation.event_type == 'agent_collision':
                    self.statistics['agent_collisions'] += 1
                elif violation.event_type == 'obstacle_collision':
                    self.statistics['obstacle_collisions'] += 1
                elif violation.event_type == 'boundary_violation':
                    self.statistics['boundary_violations'] += 1

                self.statistics['total_collisions'] += 1

            if assessment['safety_level'] == SafetyLevel.WARNING:
                self.statistics['warning_events'] += 1

    def get_safety_penalties(self, safety_assessment: Dict[int, Dict[str, Any]],
                             config: Dict[str, Any]) -> Dict[int, float]:
        """Calculate safety penalties for each agent"""
        penalties = {}

        for agent_id, assessment in safety_assessment.items():
            penalty = 0.0

            # Collision penalties
            for violation in assessment['violations']:
                if violation.severity == SafetyLevel.COLLISION:
                    penalty += config['rewards']['collision_penalty']

            # Distance-based penalties
            min_agent_distance = assessment['distances']['nearest_agent']
            min_obstacle_distance = assessment['distances']['nearest_obstacle']
            boundary_distance = assessment['distances']['boundary']

            # Agent proximity penalty
            if min_agent_distance < self.warning_distance:
                proximity_factor = max(0, (self.warning_distance - min_agent_distance) / self.warning_distance)
                penalty += config['rewards']['safety_penalty_base'] * proximity_factor ** config['rewards'][
                    'safety_penalty_multiplier']

            # Obstacle proximity penalty
            if min_obstacle_distance < self.warning_distance:
                proximity_factor = max(0, (self.warning_distance - min_obstacle_distance) / self.warning_distance)
                penalty += config['rewards']['safety_penalty_base'] * proximity_factor ** config['rewards'][
                    'safety_penalty_multiplier']

            # Boundary proximity penalty
            if boundary_distance < self.warning_distance:
                proximity_factor = max(0, (self.warning_distance - boundary_distance) / self.warning_distance)
                penalty += config['rewards']['safety_penalty_base'] * proximity_factor * 0.5

            penalties[agent_id] = penalty

        return penalties

    def is_position_safe(self, position: np.ndarray, agent_positions: List[np.ndarray],
                         agent_id: int, obstacle_manager: ObstacleManager) -> bool:
        """Check if a position is safe for an agent to move to"""
        # Check boundaries
        if (position[0] < self.agent_radius or position[0] > self.world_size[0] - self.agent_radius or
                position[1] < self.agent_radius or position[1] > self.world_size[1] - self.agent_radius):
            return False

        # Check other agents
        for i, other_position in enumerate(agent_positions):
            if i != agent_id:
                if np.linalg.norm(position - other_position) < self.min_distance:
                    return False

        # Check obstacles
        if obstacle_manager.check_collision_with_circle(position, self.agent_radius):
            return False

        return True

    def suggest_safe_action(self, current_position: np.ndarray, intended_action: np.ndarray,
                            agent_positions: List[np.ndarray], agent_id: int,
                            obstacle_manager: ObstacleManager) -> np.ndarray:
        """Suggest a safe action if the intended action would cause collision"""
        intended_position = current_position + intended_action

        if self.is_position_safe(intended_position, agent_positions, agent_id, obstacle_manager):
            return intended_action

        # Try to find a safe alternative action
        action_magnitude = np.linalg.norm(intended_action)
        if action_magnitude == 0:
            return intended_action

        # Try different angles around the intended direction
        intended_angle = np.arctan2(intended_action[1], intended_action[0])
        angle_offsets = np.linspace(-np.pi / 2, np.pi / 2, 21)  # Try 21 different angles

        for angle_offset in angle_offsets:
            new_angle = intended_angle + angle_offset
            new_action = action_magnitude * np.array([np.cos(new_angle), np.sin(new_angle)])
            new_position = current_position + new_action

            if self.is_position_safe(new_position, agent_positions, agent_id, obstacle_manager):
                return new_action

        # If no safe action found, return zero action (stop)
        return np.zeros(2)

    def reset(self):
        """Reset safety monitor for new episode"""
        self.safety_events.clear()
        self.current_violations.clear()
        self.statistics = {
            'total_collisions': 0,
            'agent_collisions': 0,
            'obstacle_collisions': 0,
            'boundary_violations': 0,
            'warning_events': 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics"""
        return self.statistics.copy()