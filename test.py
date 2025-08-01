"""
测试纯matplotlib版本的覆盖环境
"""

from coverage_env_v4 import CoverageEnvV2WithAnimation
from config_v2 import CoverageConfig


def test_matplotlib_environment():
    print("Testing Matplotlib-only Coverage Environment")
    print("=" * 50)

    # 创建配置
    config = CoverageConfig()

    # 创建环境
    env = CoverageEnvV2WithAnimation(config)

    print(f"输出目录: {env.output_dir}")
    print(f"观测维度: {env.observation_space.shape}")
    print(f"智能体数量: {env.n_agents}")

    # 重置环境
    observations, info = env.reset()
    print(f"初始覆盖率: {info['coverage']:.1f}%")

    # 运行仿真
    max_steps = 50  # 减少步数便于测试

    for step in range(max_steps):
        # 使用随机动作进行测试
        actions = []
        for i in range(env.n_agents):
            # 简单的随机策略
            action = env.action_space.sample() * 0.5  # 降低动作幅度
            actions.append(action)

        observations, rewards, terminated, truncated, info = env.step(actions)

        if step % 10 == 0 or step < 5:  # 减少输出频率
            print(f"Step {step + 1:2d}: Coverage={info['coverage']:5.1f}%, "
                  f"Rewards=[{', '.join([f'{r:6.2f}' for r in rewards])}]")

        if terminated or truncated:
            print(f"环境终止于步骤 {step + 1}")
            break

    # 关闭环境（自动保存动画）
    print("\n正在保存动画和图表...")
    env.close()

    print(f"\n仿真完成!")
    print(f"最终覆盖率: {info['coverage']:.1f}%")
    print(f"总步数: {step + 1}")
    print(f"所有文件已保存到: {env.output_dir}")
    print(f"  - PNG帧: frame_0000.png ~ frame_{env.frame_count - 1:04d}.png")
    print(f"  - GIF动画: coverage_simulation.gif")
    print(f"  - 性能图表: metrics.png")


if __name__ == "__main__":
    test_matplotlib_environment()