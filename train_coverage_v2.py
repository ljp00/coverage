"""
Complete training script for the new coverage environment
0730
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from typing import Dict, List, Any

from coverage_env_v4 import CoverageEnv
from config_v2 import CoverageConfig
from maddpg_coverage import MADDPGCoverage
from visualization import CoverageVisualizer


class CoverageTrainer:
    """Training manager for coverage environment"""

    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = CoverageConfig(config_path)

        # Create environment
        self.env = CoverageEnv(self.config)

        # Setup MADDPG
        obs_dims = [self.env.observation_space.shape[0]] * self.env.n_agents
        action_dim = self.env.action_space.shape[0]

        self.maddpg = MADDPGCoverage(
            n_agents=self.env.n_agents,
            obs_dims=obs_dims,
            action_dims=action_dim,
            lr_actor=0.0001,
            lr_critic=0.0002,
            fc1=256,
            fc2=256,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            memory_size=1000000,
            warmup_steps=1000,
            noise_scale=0.2,
            noise_decay=0.9999
        )

        # Setup visualization
        self.visualizer = CoverageVisualizer(self.env, self.config.config)

        # Training statistics
        self.training_stats = {
            'episodes': [],
            'coverage': [],
            'rewards': [],
            'steps': [],
            'safety_violations': [],
            'actor_losses': [],
            'critic_losses': []
        }

        # Create results directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.results_dir = f'results/coverage_v2_{timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)

        # Save configuration
        self.config.save_config(f'{self.results_dir}/config.json')

    def train(self, max_episodes: int = 2000, save_interval: int = 100,
              eval_interval: int = 50):
        """Main training loop"""
        print(f"Starting training for {max_episodes} episodes...")
        print(f"Results will be saved to: {self.results_dir}")

        best_coverage = 0.0

        for episode in range(max_episodes):
            episode_data = self._run_episode(episode, evaluate=False)

            # Update statistics
            self._update_stats(episode, episode_data)

            # Print progress
            if episode % 10 == 0:
                self._print_progress(episode, episode_data)

            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_data = self._run_evaluation(num_episodes=5)
                avg_coverage = np.mean(eval_data['coverage'])

                if avg_coverage > best_coverage:
                    best_coverage = avg_coverage
                    self.maddpg.save_checkpoint(episode="best")
                    print(f"New best model saved! Coverage: {avg_coverage:.2f}%")

            # Save checkpoints
            if episode % save_interval == 0 and episode > 0:
                self.maddpg.save_checkpoint(episode=episode)
                self._save_training_progress()

        # Final save
        self.maddpg.save_checkpoint(episode="final")
        self._save_training_progress()
        self._create_final_report()

        print(f"Training completed! Best coverage: {best_coverage:.2f}%")

    def _run_episode(self, episode: int, evaluate: bool = False) -> Dict[str, Any]:
        """Run a single episode"""
        observations, info = self.env.reset()

        episode_reward = 0.0
        episode_steps = 0
        trajectories = [[] for _ in range(self.env.n_agents)]
        coverage_history = []
        reward_history = []
        safety_events = []

        # Record initial state
        for i in range(self.env.n_agents):
            trajectories[i].append(self.env.agent_positions[i].copy())
        coverage_history.append(info['coverage'])

        done = False
        while not done and episode_steps < self.env.max_steps:
            # Choose actions
            actions = self.maddpg.choose_actions(observations, evaluate=evaluate)

            # Step environment
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)

            # Record trajectories
            for i in range(self.env.n_agents):
                trajectories[i].append(self.env.agent_positions[i].copy())
            coverage_history.append(info['coverage'])
            reward_history.append(np.mean(rewards))

            # Collect safety events
            if 'safety_assessment' in info:
                for agent_assessment in info['safety_assessment'].values():
                    safety_events.extend(agent_assessment.get('violations', []))

            if not evaluate:
                # Store experience and learn
                state = np.concatenate(observations)
                next_state = np.concatenate(next_observations)
                dones = [terminated] * self.env.n_agents

                self.maddpg.store_transition(observations, state, actions, rewards,
                                             next_observations, next_state, dones)
                self.maddpg.learn()

            observations = next_observations
            episode_reward += np.mean(rewards)
            episode_steps += 1
            done = terminated or truncated

        return {
            'episode': episode,
            'final_coverage': coverage_history[-1],
            'total_reward': episode_reward,
            'steps': episode_steps,
            'trajectories': trajectories,
            'coverage_history': coverage_history,
            'reward_history': reward_history,
            'safety_events': safety_events,
            'safety_violations': len(safety_events)
        }

    def _run_evaluation(self, num_episodes: int = 5) -> Dict[str, List]:
        """Run evaluation episodes"""
        eval_results = {
            'coverage': [],
            'rewards': [],
            'steps': [],
            'safety_violations': []
        }

        for _ in range(num_episodes):
            episode_data = self._run_episode(-1, evaluate=True)
            eval_results['coverage'].append(episode_data['final_coverage'])
            eval_results['rewards'].append(episode_data['total_reward'])
            eval_results['steps'].append(episode_data['steps'])
            eval_results['safety_violations'].append(episode_data['safety_violations'])

        return eval_results

    def _update_stats(self, episode: int, episode_data: Dict[str, Any]):
        """Update training statistics"""
        self.training_stats['episodes'].append(episode)
        self.training_stats['coverage'].append(episode_data['final_coverage'])
        self.training_stats['rewards'].append(episode_data['total_reward'])
        self.training_stats['steps'].append(episode_data['steps'])
        self.training_stats['safety_violations'].append(episode_data['safety_violations'])

        # Get training metrics from MADDPG
        metrics = self.maddpg.get_training_metrics()
        self.training_stats['actor_losses'].append(np.mean(metrics['actor_losses']))
        self.training_stats['critic_losses'].append(np.mean(metrics['critic_losses']))

    def _print_progress(self, episode: int, episode_data: Dict[str, Any]):
        """Print training progress"""
        print(f"Episode {episode:4d} | "
              f"Coverage: {episode_data['final_coverage']:5.1f}% | "
              f"Reward: {episode_data['total_reward']:7.2f} | "
              f"Steps: {episode_data['steps']:3d} | "
              f"Safety: {episode_data['safety_violations']:2d}")

    def _save_training_progress(self):
        """Save training progress"""
        # Save statistics
        with open(f'{self.results_dir}/training_stats.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        # Create progress plot
        self.visualizer.create_training_progress(
            self.training_stats,
            save_path=f'{self.results_dir}/training_progress.png'
        )

    def _create_final_report(self):
        """Create final training report"""
        # Calculate final statistics
        final_episodes = min(100, len(self.training_stats['coverage']))
        recent_coverage = np.mean(self.training_stats['coverage'][-final_episodes:])
        recent_rewards = np.mean(self.training_stats['rewards'][-final_episodes:])
        recent_violations = np.mean(self.training_stats['safety_violations'][-final_episodes:])

        report = {
            'training_summary': {
                'total_episodes': len(self.training_stats['episodes']),
                'final_average_coverage': float(recent_coverage),
                'final_average_reward': float(recent_rewards),
                'final_average_safety_violations': float(recent_violations),
                'best_coverage': float(max(self.training_stats['coverage'])),
                'best_reward': float(max(self.training_stats['rewards']))
            },
            'configuration': self.config.config
        }

        with open(f'{self.results_dir}/final_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total Episodes: {report['training_summary']['total_episodes']}")
        print(f"Final Avg Coverage: {recent_coverage:.1f}%")
        print(f"Final Avg Reward: {recent_rewards:.2f}")
        print(f"Best Coverage: {max(self.training_stats['coverage']):.1f}%")
        print(f"Safety Violations: {recent_violations:.1f}/episode")
        print("=" * 50)


def main():
    """Main training function"""
    # You can specify a custom config file here
    config_path = None  # or "path/to/your/config.json"

    trainer = CoverageTrainer(config_path)
    trainer.train(max_episodes=2000, save_interval=100, eval_interval=50)


if __name__ == "__main__":
    main()