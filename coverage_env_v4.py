"""
Coverage Environment V2 with Animation and Frame Saving
带有动画保存功能的覆盖环境
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.backends.backend_agg as agg
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import json
import copy
import os
from datetime import datetime
from PIL import Image
import io


class CoverageEnv:
    """带有动画保存功能的覆盖环境"""

    def __init__(self, config):
        self.config = config.config if hasattr(config, 'config') else config

        # 环境参数
        self.world_size = tuple(self.config['environment']['size'])
        self.grid_resolution = self.config['environment']['grid_resolution']
        self.max_steps = self.config['environment']['max_steps']

        # 智能体参数
        self.n_agents = self.config['agents']['n_agents']
        self.sensor_range = self.config['agents']['sensor_range']
        self.min_distance = self.config['agents']['min_distance']
        self.max_velocity = self.config['agents']['max_velocity']
        self.agent_radius = self.config['agents']['agent_radius']

        # 激光传感器参数
        self.num_lasers = 16
        self.laser_range = 2.0
        self.monitor_radius = 0.2

        # 激光传感器状态
        self.multi_current_lasers = [[self.laser_range for _ in range(self.num_lasers)]
                                     for _ in range(self.n_agents)]

        # 覆盖网格
        self.grid_size_x = int(self.world_size[0] * self.grid_resolution / 10.0)
        self.grid_size_y = int(self.world_size[1] * self.grid_resolution / 10.0)
        self.coverage_grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=bool)

        # 动作和观测空间
        self.action_space = spaces.Box(
            low=np.array([-self.max_velocity, -self.max_velocity]),
            high=np.array([self.max_velocity, self.max_velocity]),
            dtype=np.float32
        )

        obs_dim = (
                4 +  # 自身状态
                (self.n_agents - 1) * 2 +  # 其他智能体
                self.num_lasers +  # 激光束
                3 +  # 局部覆盖信息
                4 +  # 边界距离
                2  # 任务状态
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 动画和可视化相关
        self.enable_animation = True
        self.animation_frames = []
        self.output_dir = None
        self.frame_count = 0

        # 历史轨迹记录
        self.agent_trajectories = [[] for _ in range(self.n_agents)]
        self.coverage_history = []
        self.reward_history = []

        # 初始化其他组件
        self._init_obstacles()
        self._init_agents()
        self._setup_animation()

        self.step_count = 0
        self.total_coverage_area = self.grid_size_x * self.grid_size_y

        print(f"带动画保存的覆盖环境初始化完成:")
        print(f"  观测维度: {obs_dim}")
        print(f"  激光束数量: {self.num_lasers}")
        print(f"  智能体数量: {self.n_agents}")

    def _setup_animation(self):
        """设置动画保存"""
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"coverage_simulation_{timestamp}"

        if self.enable_animation:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            print(f"动画输出目录: {self.output_dir}")

        # 颜色配置
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

    def _init_obstacles(self):
        """初始化障碍物"""
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

    def _init_agents(self):
        """初始化智能体"""
        self.agent_positions = []
        self.agent_velocities = []

        for i in range(self.n_agents):
            position = self._generate_safe_position()
            self.agent_positions.append(position)
            self.agent_velocities.append(np.zeros(2, dtype=np.float32))
            self.agent_trajectories[i] = [position.copy()]

    def _generate_safe_position(self) -> np.ndarray:
        """生成安全的初始位置"""
        max_attempts = 100

        for _ in range(max_attempts):
            position = np.array([
                np.random.uniform(self.agent_radius, self.world_size[0] - self.agent_radius),
                np.random.uniform(self.agent_radius, self.world_size[1] - self.agent_radius)
            ])

            # 检查与其他智能体的距离
            safe_from_agents = True
            for existing_pos in self.agent_positions:
                if np.linalg.norm(position - existing_pos) < self.min_distance:
                    safe_from_agents = False
                    break

            # 检查与障碍物的距离
            safe_from_obstacles = True
            for obstacle in self.all_obstacles:
                obs_pos = obstacle['current_position'] if obstacle['is_dynamic'] else obstacle['position']
                distance = np.linalg.norm(position - obs_pos)
                if distance < (obstacle['radius'] + self.agent_radius + 0.1):
                    safe_from_obstacles = False
                    break

            if safe_from_agents and safe_from_obstacles:
                return position

        return np.array([1.0, 1.0])

    def render_frame(self, save_frame=True):
        """渲染当前帧并可选保存"""
        try:
            # 创建matplotlib图形
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, self.world_size[0])
            ax.set_ylim(0, self.world_size[1])
            ax.set_aspect('equal')

            # 绘制覆盖网格
            self._draw_coverage_grid(ax)

            # 绘制障碍物
            self._draw_obstacles(ax)

            # 绘制智能体和相关信息
            self._draw_agents(ax)

            # 添加信息面板
            self._add_info_panel(ax)

            # 设置标题和标签
            coverage_rate = self.get_coverage_percentage()
            ax.set_title(f'Multi-Agent Coverage Control\n'
                         f'Step: {self.step_count}, Coverage: {coverage_rate:.1f}%',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)

            # 添加网格
            ax.grid(True, alpha=0.3)

            # 保存帧
            if save_frame and self.enable_animation:
                # 保存为PNG
                frame_filename = f"frame_{self.frame_count:04d}.png"
                frame_path = os.path.join(self.output_dir, frame_filename)
                plt.savefig(frame_path, dpi=150, bbox_inches='tight',
                            facecolor='white', edgecolor='none')

                # 保存到内存用于GIF
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                buf.seek(0)
                img = Image.open(buf)
                self.animation_frames.append(img.copy())
                buf.close()

                self.frame_count += 1

            # 转换为numpy数组用于返回
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf)

            plt.close(fig)
            return image_array

        except Exception as e:
            print(f"渲染帧时出错: {e}")
            # 返回空白图像
            return np.zeros((800, 800, 4), dtype=np.uint8)

    def _draw_coverage_grid(self, ax):
        """绘制覆盖网格"""
        cell_width = self.world_size[0] / self.grid_size_x
        cell_height = self.world_size[1] / self.grid_size_y

        for gx in range(self.grid_size_x):
            for gy in range(self.grid_size_y):
                x = gx * cell_width
                y = gy * cell_height

                color = self.colors['covered_area'] if self.coverage_grid[gx, gy] else self.colors['uncovered_area']
                rect = patches.Rectangle((x, y), cell_width, cell_height,
                                         facecolor=color, edgecolor=self.colors['grid_lines'],
                                         linewidth=0.1, alpha=0.7)
                ax.add_patch(rect)

    def _draw_obstacles(self, ax):
        """绘制障碍物"""
        # 静态障碍物
        for obstacle in self.static_obstacles:
            circle = patches.Circle(obstacle['position'], obstacle['radius'],
                                    facecolor=self.colors['static_obstacle'],
                                    edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(circle)

            # 添加标签
            ax.text(obstacle['position'][0], obstacle['position'][1], 'S',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # 动态障碍物
        for obstacle in self.dynamic_obstacles:
            circle = patches.Circle(obstacle['current_position'], obstacle['radius'],
                                    facecolor=self.colors['dynamic_obstacle'],
                                    edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(circle)

            # 添加标签
            ax.text(obstacle['current_position'][0], obstacle['current_position'][1], 'D',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')

            # 绘制运动轨迹（振荡运动）
            if obstacle['type'] == 'oscillating':
                ax.plot([obstacle['initial_position'][0], obstacle['current_position'][0]],
                        [obstacle['initial_position'][1], obstacle['current_position'][1]],
                        'k--', alpha=0.5, linewidth=1)

    def _draw_agents(self, ax):
        """绘制智能体及相关信息"""
        for i in range(self.n_agents):
            pos = self.agent_positions[i]
            vel = self.agent_velocities[i]
            color = self.colors['agents'][i % len(self.colors['agents'])]

            # 绘制轨迹
            if len(self.agent_trajectories[i]) > 1:
                trajectory = np.array(self.agent_trajectories[i])
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                        color=self.colors['trajectories'][i % len(self.colors['trajectories'])],
                        alpha=0.6, linewidth=2, label=f'Agent {i} trajectory')

            # 绘制传感器范围
            sensor_circle = patches.Circle(pos, self.sensor_range,
                                           facecolor=self.colors['sensor_range'],
                                           edgecolor=color, linewidth=1.5, alpha=0.3)
            ax.add_patch(sensor_circle)

            # 绘制安全区域
            safety_circle = patches.Circle(pos, self.min_distance,
                                           facecolor=self.colors['safety_zone'],
                                           edgecolor='orange', linewidth=1, alpha=0.3)
            ax.add_patch(safety_circle)

            # 绘制覆盖范围
            coverage_circle = patches.Circle(pos, self.monitor_radius,
                                             facecolor='blue', alpha=0.1,
                                             edgecolor='blue', linewidth=1)
            ax.add_patch(coverage_circle)

            # 绘制激光束（可选，显示几条主要方向）
            if hasattr(self, 'show_lasers') and self.show_lasers:
                self._draw_laser_beams(ax, i, pos)

            # 绘制智能体本体
            agent_circle = patches.Circle(pos, self.agent_radius,
                                          facecolor=color, edgecolor='black',
                                          linewidth=2, alpha=0.9)
            ax.add_patch(agent_circle)

            # 绘制速度向量
            if np.linalg.norm(vel) > 0.01:
                ax.arrow(pos[0], pos[1], vel[0] * 0.5, vel[1] * 0.5,
                         head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.8)

            # 添加智能体ID
            ax.text(pos[0], pos[1], str(i), ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')

    def _draw_laser_beams(self, ax, agent_idx, pos):
        """绘制激光束（显示主要方向）"""
        if agent_idx < len(self.multi_current_lasers):
            lasers = self.multi_current_lasers[agent_idx]
            # 只显示4个主要方向的激光束
            main_directions = [0, 4, 8, 12]  # 对应0°, 90°, 180°, 270°

            for laser_idx in main_directions:
                if laser_idx < len(lasers):
                    angle = laser_idx * 2 * np.pi / self.num_lasers
                    laser_length = lasers[laser_idx]

                    end_x = pos[0] + laser_length * np.cos(angle)
                    end_y = pos[1] + laser_length * np.sin(angle)

                    ax.plot([pos[0], end_x], [pos[1], end_y],
                            color=self.colors['laser_beam'], linewidth=1, alpha=0.6)

    def _add_info_panel(self, ax):
        """添加信息面板"""
        info_text = []

        # 覆盖率信息
        coverage_rate = self.get_coverage_percentage()
        info_text.append(f"Coverage: {coverage_rate:.1f}%")

        # 步数信息
        info_text.append(f"Step: {self.step_count}/{self.max_steps}")

        # 智能体状态
        for i in range(self.n_agents):
            vel_magnitude = np.linalg.norm(self.agent_velocities[i])
            info_text.append(f"Agent {i}: v={vel_magnitude:.2f}")

        # 在图的右上角显示信息
        info_str = '\n'.join(info_text)
        ax.text(0.98, 0.98, info_str, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, fontfamily='monospace')

    def save_animation(self, filename_prefix="coverage_simulation"):
        """保存GIF动画"""
        if not self.animation_frames:
            print("没有帧数据可保存")
            return None

        try:
            gif_filename = f"{filename_prefix}.gif"
            gif_path = os.path.join(self.output_dir, gif_filename)

            # 保存GIF
            self.animation_frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.animation_frames[1:],
                duration=200,  # 每帧200ms
                loop=0
            )

            print(f"GIF动画已保存: {gif_path}")
            print(f"总共 {len(self.animation_frames)} 帧")
            print(f"单独的PNG帧已保存在: {self.output_dir}")

            return gif_path

        except Exception as e:
            print(f"保存GIF时出错: {e}")
            return None

    def save_metrics(self):
        """保存性能指标"""
        if not self.output_dir:
            return

        try:
            # 保存覆盖率和奖励历史
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # 覆盖率历史
            ax1.plot(self.coverage_history, 'b-', linewidth=2)
            ax1.set_title('Coverage Rate History')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Coverage Rate (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)

            # 奖励历史
            if self.reward_history:
                rewards_array = np.array(self.reward_history)
                for i in range(self.n_agents):
                    ax2.plot(rewards_array[:, i], label=f'Agent {i}', linewidth=2)
                ax2.set_title('Reward History')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Reward')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            metrics_path = os.path.join(self.output_dir, 'metrics.png')
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"性能指标图已保存: {metrics_path}")

        except Exception as e:
            print(f"保存性能指标时出错: {e}")

    # 其他方法保持不变...
    def update_lasers(self, agent_idx: int):
        """更新激光传感器"""
        agent_pos = self.agent_positions[agent_idx]
        current_lasers = [self.laser_range] * self.num_lasers
        collision_detected = False

        laser_angles = np.linspace(0, 2 * np.pi, self.num_lasers, endpoint=False)

        for laser_idx, angle in enumerate(laser_angles):
            laser_dir = np.array([np.cos(angle), np.sin(angle)])
            min_distance = self.laser_range

            # 检查障碍物
            for obstacle in self.all_obstacles:
                obs_pos = obstacle['current_position'] if obstacle['is_dynamic'] else obstacle['position']
                obs_radius = obstacle['radius']

                intersection_dist = self._ray_circle_intersection(
                    agent_pos, laser_dir, obs_pos, obs_radius
                )

                if intersection_dist is not None and intersection_dist < min_distance:
                    min_distance = intersection_dist
                    if min_distance < self.agent_radius + 0.01:
                        collision_detected = True

            # 检查其他智能体
            for other_idx, other_pos in enumerate(self.agent_positions):
                if other_idx != agent_idx:
                    intersection_dist = self._ray_circle_intersection(
                        agent_pos, laser_dir, other_pos, self.agent_radius
                    )

                    if intersection_dist is not None and intersection_dist < min_distance:
                        min_distance = intersection_dist
                        if min_distance < self.min_distance:
                            collision_detected = True

            # 检查边界
            boundary_dist = self._ray_boundary_intersection(agent_pos, laser_dir)
            if boundary_dist < min_distance:
                min_distance = boundary_dist

            current_lasers[laser_idx] = min_distance

        self.multi_current_lasers[agent_idx] = current_lasers
        return collision_detected

    def _ray_circle_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                                 circle_center: np.ndarray, circle_radius: float) -> Optional[float]:
        """射线与圆的交点计算"""
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

        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None

    def _ray_boundary_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> float:
        """射线与边界的交点计算"""
        distances = []

        # 四个边界
        if ray_dir[0] != 0:
            t = -ray_origin[0] / ray_dir[0]
            if t > 0:
                y = ray_origin[1] + t * ray_dir[1]
                if 0 <= y <= self.world_size[1]:
                    distances.append(t)

            t = (self.world_size[0] - ray_origin[0]) / ray_dir[0]
            if t > 0:
                y = ray_origin[1] + t * ray_dir[1]
                if 0 <= y <= self.world_size[1]:
                    distances.append(t)

        if ray_dir[1] != 0:
            t = -ray_origin[1] / ray_dir[1]
            if t > 0:
                x = ray_origin[0] + t * ray_dir[0]
                if 0 <= x <= self.world_size[0]:
                    distances.append(t)

            t = (self.world_size[1] - ray_origin[1]) / ray_dir[1]
            if t > 0:
                x = ray_origin[0] + t * ray_dir[0]
                if 0 <= x <= self.world_size[0]:
                    distances.append(t)

        return min(distances) if distances else self.laser_range

    def step(self, actions):
        """环境步进"""
        # 更新动态障碍物
        self._update_dynamic_obstacles()

        # 更新智能体位置
        for i, action in enumerate(actions):
            action = np.clip(action, -self.max_velocity, self.max_velocity)

            self.agent_velocities[i] = action
            new_position = self.agent_positions[i] + action

            # 边界约束
            new_position[0] = np.clip(new_position[0], self.agent_radius,
                                      self.world_size[0] - self.agent_radius)
            new_position[1] = np.clip(new_position[1], self.agent_radius,
                                      self.world_size[1] - self.agent_radius)

            self.agent_positions[i] = new_position
            self.agent_trajectories[i].append(new_position.copy())

            # 限制轨迹长度
            if len(self.agent_trajectories[i]) > 500:
                self.agent_trajectories[i] = self.agent_trajectories[i][-500:]

        # 更新激光传感器
        collisions = [self.update_lasers(i) for i in range(self.n_agents)]

        # 更新覆盖网格
        self._update_coverage_grid()

        # 计算奖励
        rewards = self._compute_rewards(collisions)

        # 记录历史
        self.coverage_history.append(self.get_coverage_percentage())
        self.reward_history.append(rewards.copy())

        # 渲染当前帧
        if self.enable_animation:
            self.render_frame(save_frame=True)

        # 检查终止条件
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        # 获取观测
        observations = self._get_observations()

        info = {
            'coverage': self.get_coverage_percentage(),
            'collisions': collisions,
            'step_count': self.step_count
        }

        return observations, rewards, terminated, truncated, info

    def _update_dynamic_obstacles(self):
        """更新动态障碍物"""
        for obstacle in self.dynamic_obstacles:
            if obstacle['type'] == 'oscillating':
                time_factor = self.step_count * 0.1
                displacement = obstacle['amplitude'] * np.sin(
                    obstacle['frequency'] * time_factor + obstacle['phase']
                )
                obstacle['current_position'] = obstacle['initial_position'] + displacement

                # 边界检查
                obstacle['current_position'][0] = np.clip(
                    obstacle['current_position'][0],
                    obstacle['radius'],
                    self.world_size[0] - obstacle['radius']
                )
                obstacle['current_position'][1] = np.clip(
                    obstacle['current_position'][1],
                    obstacle['radius'],
                    self.world_size[1] - obstacle['radius']
                )

    def _update_coverage_grid(self):
        """更新覆盖网格"""
        for i, agent_pos in enumerate(self.agent_positions):
            for gx in range(self.grid_size_x):
                for gy in range(self.grid_size_y):
                    grid_world_x = gx * self.world_size[0] / self.grid_size_x
                    grid_world_y = gy * self.world_size[1] / self.grid_size_y
                    grid_pos = np.array([grid_world_x, grid_world_y])

                    distance = np.linalg.norm(agent_pos - grid_pos)
                    if distance <= self.monitor_radius:
                        self.coverage_grid[gx, gy] = True

    def get_coverage_percentage(self) -> float:
        """获取覆盖百分比"""
        covered_cells = np.sum(self.coverage_grid)
        return (covered_cells / self.total_coverage_area) * 100.0

    def _get_observations(self):
        """获取观测"""
        observations = []

        for i in range(self.n_agents):
            obs = []

            # 自身状态
            pos = self.agent_positions[i]
            vel = self.agent_velocities[i]
            obs.extend([
                pos[0] / self.world_size[0],
                pos[1] / self.world_size[1],
                vel[0] / self.max_velocity,
                vel[1] / self.max_velocity
            ])

            # 其他智能体相对位置
            for j in range(self.n_agents):
                if j != i:
                    other_pos = self.agent_positions[j]
                    relative_pos = other_pos - pos
                    obs.extend([
                        relative_pos[0] / self.world_size[0],
                        relative_pos[1] / self.world_size[1]
                    ])

            # 激光束读数
            lasers = self.multi_current_lasers[i]
            for laser_dist in lasers:
                obs.append(laser_dist / self.laser_range)

            # 局部覆盖信息
            local_coverage = self._compute_local_coverage(pos)
            coverage_gradient = self._compute_coverage_gradient(pos)
            obs.extend([local_coverage, coverage_gradient[0], coverage_gradient[1]])

            # 边界距离
            boundary_distances = [
                pos[0] / self.world_size[0],
                (self.world_size[0] - pos[0]) / self.world_size[0],
                pos[1] / self.world_size[1],
                (self.world_size[1] - pos[1]) / self.world_size[1]
            ]
            obs.extend(boundary_distances)

            # 任务状态
            global_coverage = self.get_coverage_percentage() / 100.0
            step_progress = self.step_count / self.max_steps
            obs.extend([global_coverage, step_progress])

            # 验证维度
            expected_dim = self.observation_space.shape[0]
            if len(obs) != expected_dim:
                if len(obs) < expected_dim:
                    obs.extend([0.0] * (expected_dim - len(obs)))
                else:
                    obs = obs[:expected_dim]

            observations.append(np.array(obs, dtype=np.float32))

        return observations

    def _compute_local_coverage(self, agent_pos: np.ndarray) -> float:
        """计算局部覆盖率"""
        local_radius = 1.0
        total_points = 0
        covered_points = 0

        for dx in np.linspace(-local_radius, local_radius, 10):
            for dy in np.linspace(-local_radius, local_radius, 10):
                if dx * dx + dy * dy <= local_radius * local_radius:
                    sample_pos = agent_pos + np.array([dx, dy])

                    if (0 <= sample_pos[0] <= self.world_size[0] and
                            0 <= sample_pos[1] <= self.world_size[1]):
                        total_points += 1

                        gx = int(sample_pos[0] / self.world_size[0] * self.grid_size_x)
                        gy = int(sample_pos[1] / self.world_size[1] * self.grid_size_y)
                        gx = np.clip(gx, 0, self.grid_size_x - 1)
                        gy = np.clip(gy, 0, self.grid_size_y - 1)

                        if self.coverage_grid[gx, gy]:
                            covered_points += 1

        return covered_points / max(total_points, 1)

    def _compute_coverage_gradient(self, agent_pos: np.ndarray) -> np.ndarray:
        """计算覆盖梯度"""
        eps = 0.1

        pos_x_plus = agent_pos + np.array([eps, 0])
        pos_x_minus = agent_pos - np.array([eps, 0])
        coverage_x_plus = self._compute_local_coverage(pos_x_plus)
        coverage_x_minus = self._compute_local_coverage(pos_x_minus)
        grad_x = (coverage_x_plus - coverage_x_minus) / (2 * eps)

        pos_y_plus = agent_pos + np.array([0, eps])
        pos_y_minus = agent_pos - np.array([0, eps])
        coverage_y_plus = self._compute_local_coverage(pos_y_plus)
        coverage_y_minus = self._compute_local_coverage(pos_y_minus)
        grad_y = (coverage_y_plus - coverage_y_minus) / (2 * eps)

        return np.array([grad_x, grad_y])

    def _compute_rewards(self, collisions):
        """计算奖励"""
        rewards = []

        for i in range(self.n_agents):
            reward = 0.0

            # 覆盖奖励
            coverage_reward = self.get_coverage_percentage() / 100.0
            reward += coverage_reward * 10.0

            # 碰撞惩罚
            if collisions[i]:
                reward -= 20.0

            # 安全奖励
            min_laser = min(self.multi_current_lasers[i])
            safety_reward = min_laser / self.laser_range
            reward += safety_reward * 2.0

            rewards.append(reward)

        return np.array(rewards)

    def reset(self, seed=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        # 重置状态
        self.coverage_grid.fill(False)
        self.agent_positions = []
        self.agent_velocities = []
        self.agent_trajectories = [[] for _ in range(self.n_agents)]
        self.coverage_history = []
        self.reward_history = []
        self.animation_frames = []
        self.frame_count = 0
        self.step_count = 0

        # 重新初始化智能体
        for i in range(self.n_agents):
            position = self._generate_safe_position()
            self.agent_positions.append(position)
            self.agent_velocities.append(np.zeros(2, dtype=np.float32))
            self.agent_trajectories[i] = [position.copy()]

        # 初始化激光传感器
        for i in range(self.n_agents):
            self.update_lasers(i)

        # 渲染初始帧
        if self.enable_animation:
            self.render_frame(save_frame=True)

        observations = self._get_observations()

        info = {
            'coverage': self.get_coverage_percentage(),
            'step_count': self.step_count
        }

        return observations, info

    def close(self):
        """关闭环境并保存动画"""
        if self.enable_animation and self.animation_frames:
            self.save_animation()
            self.save_metrics()
        plt.close('all')