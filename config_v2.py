"""
Configuration management for multi-agent coverage environment v2
"""

import json
import os
from typing import Dict, List, Tuple, Any


class CoverageConfig:
    """Configuration manager for coverage environment"""

    def __init__(self, config_path: str = None):
        """Initialize configuration with default values"""
        self.config = self._get_default_config()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'environment': {
                'size': [10.0, 10.0],
                'grid_resolution': 50,
                'max_steps': 500,
                'time_limit': 300.0
            },
            'agents': {
                'n_agents': 3,
                'sensor_range': 1.5,
                'min_distance': 0.5,
                'max_velocity': 0.5,
                'agent_radius': 0.2
            },
            'obstacles': {
                'static_obstacles': [
                    {'position': [2.5, 2.5], 'radius': 0.7},
                    {'position': [7.5, 7.5], 'radius': 0.7}
                ],
                'dynamic_obstacles': [
                    {
                        'type': 'oscillating',
                        'position': [5.0, 5.0],
                        'radius': 0.5,
                        'amplitude': [2.0, 0.0],
                        'frequency': [0.02, 0.0],
                        'phase': [0.0, 0.0]
                    }
                ]
            },
            'rewards': {
                'new_coverage': 10.0,
                'overlap_penalty_base': -0.5,
                'overlap_penalty_multiplier': 0.3,
                'collision_penalty': -10.0,
                'safety_penalty_base': -1.0,
                'safety_penalty_multiplier': 2.0,
                'time_penalty': -0.01,
                'energy_penalty_factor': 0.05,
                'efficiency_bonus': 1.0
            },
            'safety': {
                'warning_distance_multiplier': 1.5,
                'danger_distance_multiplier': 1.2,
                'collision_threshold': 0.9
            },
            'visualization': {
                'figsize': [12, 10],
                'dpi': 200,
                'animation_fps': 10,
                'colormap': 'Blues'
            }
        }

    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        self._update_config(self.config, loaded_config)

    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _update_config(self, base_config: Dict, update_config: Dict):
        """Recursively update configuration"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'agents.sensor_range')"""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value