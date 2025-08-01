"""
Advanced visualization tools for multi-agent coverage environment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from typing import List, Dict, Any, Tuple
import os


class CoverageVisualizer:
    """Advanced visualization for coverage environment"""

    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.fig = None
        self.axes = None

    def create_episode_summary(self, episode_data: Dict[str, Any],
                               save_path: str = None) -> plt.Figure:
        """Create comprehensive episode summary visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Extract data
        trajectories = episode_data['trajectories']
        coverage_history = episode_data['coverage_history']
        reward_history = episode_data['reward_history']
        safety_events = episode_data.get('safety_events', [])

        # 1. Agent trajectories
        self._plot_trajectories(axes[0, 0], trajectories)

        # 2. Final coverage map
        self._plot_coverage_map(axes[0, 1])

        # 3. Coverage progress
        self._plot_coverage_progress(axes[0, 2], coverage_history)

        # 4. Reward components
        self._plot_reward_components(axes[1, 0], reward_history)

        # 5. Safety analysis
        self._plot_safety_analysis(axes[1, 1], safety_events)

        # 6. Performance metrics
        self._plot_performance_metrics(axes[1, 2], episode_data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_trajectories(self, ax, trajectories):
        """Plot agent trajectories"""
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))

        # Environment boundaries
        ax.plot([0, self.env.world_size[0], self.env.world_size[0], 0, 0],
                [0, 0, self.env.world_size[1], self.env.world_size[1], 0],
                'k-', linewidth=2)

        # Obstacles
        for obs in self.env.obstacle_manager.static_obstacles:
            circle = Circle(obs.position, obs.radius, color='gray', alpha=0.7)
            ax.add_patch(circle)

        for obs in self.env.obstacle_manager.dynamic_obstacles:
            circle = Circle(obs.position, obs.radius, color='red', alpha=0.7)
            ax.add_patch(circle)

        # Agent trajectories
        for i, traj in enumerate(trajectories):
            traj_array = np.array(traj)
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                    color=colors[i], linewidth=2, alpha=0.7, label=f'Agent {i + 1}')
            ax.plot(traj_array[0, 0], traj_array[0, 1],
                    'o', color=colors[i], markersize=8)  # Start
            ax.plot(traj_array[-1, 0], traj_array[-1, 1],
                    's', color=colors[i], markersize=8)  # End

        ax.set_title('Agent Trajectories')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    def _plot_coverage_map(self, ax):
        """Plot final coverage map"""
        coverage_img = ax.imshow(self.env.coverage_grid.T,
                                 origin='lower',
                                 extent=(0, self.env.world_size[0], 0, self.env.world_size[1]),
                                 cmap='Blues', vmin=0, vmax=1)

        # Add final agent positions
        colors = plt.cm.rainbow(np.linspace(0, 1, self.env.n_agents))
        for i, pos in enumerate(self.env.agent_positions):
            ax.plot(pos[0], pos[1], 'o', color=colors[i], markersize=10)
            circle = Circle(pos, self.env.sensor_range,
                            color=colors[i], fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

        ax.set_title(f'Final Coverage: {self.env.get_coverage_percentage():.1f}%')
        plt.colorbar(coverage_img, ax=ax, label='Coverage')

    def _plot_coverage_progress(self, ax, coverage_history):
        """Plot coverage progress over time"""
        ax.plot(coverage_history, 'g-', linewidth=2)
        ax.set_title('Coverage Progress')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Coverage Ratio')
        ax.grid(True)
        ax.set_ylim(0, 1)

    def _plot_reward_components(self, ax, reward_history):
        """Plot reward components"""
        # This assumes reward_history contains component breakdowns
        if isinstance(reward_history[0], dict):
            components = list(reward_history[0].keys())
            for component in components:
                values = [r[component] for r in reward_history]
                ax.plot(values, label=component, alpha=0.7)
        else:
            ax.plot(reward_history, 'b-', linewidth=2)

        ax.set_title('Reward Components')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)

    def _plot_safety_analysis(self, ax, safety_events):
        """Plot safety analysis"""
        event_types = ['agent_collision', 'obstacle_collision', 'boundary_violation']
        counts = [sum(1 for e in safety_events if e.event_type == et) for et in event_types]

        bars = ax.bar(event_types, counts, color=['red', 'orange', 'yellow'])
        ax.set_title('Safety Violations')
        ax.set_ylabel('Count')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

    def _plot_performance_metrics(self, ax, episode_data):
        """Plot performance metrics"""
        metrics = {
            'Final Coverage': episode_data.get('final_coverage', 0) * 100,
            'Steps Taken': episode_data.get('steps_taken', 0),
            'Safety Score': 100 - episode_data.get('safety_violations', 0) * 10,
            'Efficiency': episode_data.get('efficiency_score', 0) * 100
        }

        bars = ax.bar(metrics.keys(), metrics.values(),
                      color=['green', 'blue', 'orange', 'purple'])
        ax.set_title('Performance Metrics')
        ax.set_ylabel('Score')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom')

    def create_training_progress(self, training_data: Dict[str, List],
                                 save_path: str = None) -> plt.Figure:
        """Create training progress visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        episodes = range(len(training_data['coverage']))

        # Coverage progress
        axes[0, 0].plot(episodes, training_data['coverage'], 'g-', alpha=0.7)
        axes[0, 0].set_title('Coverage Progress During Training')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Final Coverage (%)')
        axes[0, 0].grid(True)

        # Reward progress
        axes[0, 1].plot(episodes, training_data['rewards'], 'b-', alpha=0.7)
        axes[0, 1].set_title('Reward Progress During Training')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Reward')
        axes[0, 1].grid(True)

        # Safety metrics
        axes[1, 0].plot(episodes, training_data['safety_violations'], 'r-', alpha=0.7)
        axes[1, 0].set_title('Safety Violations During Training')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Violations per Episode')
        axes[1, 0].grid(True)

        # Learning curves (losses)
        if 'actor_losses' in training_data:
            axes[1, 1].plot(episodes, training_data['actor_losses'], 'purple',
                            alpha=0.7, label='Actor Loss')
            axes[1, 1].plot(episodes, training_data['critic_losses'], 'orange',
                            alpha=0.7, label='Critic Loss')
            axes[1, 1].set_title('Learning Curves')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_animation(self, trajectory_data: List[List[np.ndarray]],
                         coverage_data: List[np.ndarray],
                         save_path: str = None) -> animation.FuncAnimation:
        """Create animated visualization of episode"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Setup axes
        for ax in [ax1, ax2]:
            ax.set_xlim(0, self.env.world_size[0])
            ax.set_ylim(0, self.env.world_size[1])
            ax.set_aspect('equal')
            ax.grid(True)

        ax1.set_title('Agent Movement')
        ax2.set_title('Coverage Progress')

        # Initialize plot elements
        colors = plt.cm.rainbow(np.linspace(0, 1, self.env.n_agents))

        # Agent markers and trails
        agent_markers = []
        agent_trails = []
        sensor_circles = []

        for i in range(self.env.n_agents):
            marker, = ax1.plot([], [], 'o', color=colors[i], markersize=8)
            trail, = ax1.plot([], [], '-', color=colors[i], alpha=0.5)
            sensor = Circle((0, 0), self.env.sensor_range,
                            color=colors[i], fill=False, alpha=0.3)
            ax1.add_patch(sensor)

            agent_markers.append(marker)
            agent_trails.append(trail)
            sensor_circles.append(sensor)

        # Coverage visualization
        coverage_img = ax2.imshow(np.zeros_like(coverage_data[0]).T,
                                  origin='lower',
                                  extent=(0, self.env.world_size[0], 0, self.env.world_size[1]),
                                  cmap='Blues', vmin=0, vmax=1)

        # Add obstacles to both plots
        for ax in [ax1, ax2]:
            for obs in self.env.obstacle_manager.static_obstacles:
                circle = Circle(obs.position, obs.radius, color='gray', alpha=0.7)
                ax.add_patch(circle)

        coverage_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                                 fontsize=12, verticalalignment='top')

        def animate(frame):
            # Update agent positions and trails
            for i in range(self.env.n_agents):
                if frame < len(trajectory_data[i]):
                    pos = trajectory_data[i][frame]
                    agent_markers[i].set_data([pos[0]], [pos[1]])
                    sensor_circles[i].center = pos

                    # Update trail
                    trail_data = np.array(trajectory_data[i][:frame + 1])
                    if len(trail_data) > 0:
                        agent_trails[i].set_data(trail_data[:, 0], trail_data[:, 1])

            # Update coverage
            if frame < len(coverage_data):
                coverage_img.set_array(coverage_data[frame].T)
                coverage_pct = np.mean(coverage_data[frame]) * 100
                coverage_text.set_text(f'Coverage: {coverage_pct:.1f}%')

            return agent_markers + agent_trails + sensor_circles + [coverage_img, coverage_text]

        # Create animation
        max_frames = max(len(traj) for traj in trajectory_data)
        anim = animation.FuncAnimation(fig, animate, frames=max_frames,
                                       interval=100, blit=True, repeat=True)

        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=10, dpi=200)

        return anim