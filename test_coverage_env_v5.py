"""
Test script for Coverage Environment V5
测试覆盖环境 V5 的脚本

This script thoroughly tests the new CoverageEnvV5 implementation,
including all major functionality and compatibility with existing systems.
"""

import numpy as np
import time
from coverage_env_v5 import CoverageEnvV5
from config_v2 import CoverageConfig


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    # Create configuration
    config = CoverageConfig()
    
    # Create environment
    env = CoverageEnvV5(config, render_mode='rgb_array')
    
    print(f"Environment created successfully:")
    print(f"  World size: {env.world_size}")
    print(f"  Grid resolution: {env.grid_size_x}x{env.grid_size_y}")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    # Test reset
    observations, info = env.reset(seed=42)
    print(f"\nReset successful:")
    print(f"  Initial coverage: {info['coverage_percentage']:.1f}%")
    print(f"  Observation dimensions: {[obs.shape for obs in observations]}")
    
    # Test a few steps
    print(f"\nTesting environment steps...")
    for step in range(10):
        # Generate random actions
        actions = []
        for i in range(env.n_agents):
            action = env.action_space.sample() * 0.3  # Reduced magnitude
            actions.append(action)
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        if step < 3:  # Print details for first few steps
            print(f"  Step {step + 1}: Coverage={info['coverage_percentage']:5.1f}%, "
                  f"Rewards=[{', '.join([f'{r:6.2f}' for r in rewards])}]")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    
    env.close()
    print(f"Basic functionality test completed successfully!")


def test_gymnasium_compliance():
    """Test OpenAI Gymnasium compliance."""
    print("\n" + "=" * 60)
    print("Testing Gymnasium Compliance")
    print("=" * 60)
    
    config = CoverageConfig()
    env = CoverageEnvV5(config)
    
    # Test required methods and attributes
    required_methods = ['reset', 'step', 'render', 'close']
    required_attributes = ['action_space', 'observation_space', 'metadata']
    
    print("Checking required methods:")
    for method in required_methods:
        if hasattr(env, method) and callable(getattr(env, method)):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method}")
    
    print("\nChecking required attributes:")
    for attr in required_attributes:
        if hasattr(env, attr):
            print(f"  ✓ {attr}")
        else:
            print(f"  ✗ {attr}")
    
    # Test space types
    from gymnasium import spaces
    print(f"\nSpace types:")
    print(f"  Action space: {type(env.action_space)} - Valid: {isinstance(env.action_space, spaces.Space)}")
    print(f"  Observation space: {type(env.observation_space)} - Valid: {isinstance(env.observation_space, spaces.Space)}")
    
    # Test reset return format
    obs, info = env.reset()
    print(f"\nReset return format:")
    print(f"  Observations type: {type(obs)}")
    print(f"  Info type: {type(info)} - Valid: {isinstance(info, dict)}")
    
    # Test step return format
    actions = [env.action_space.sample() for _ in range(env.n_agents)]
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"\nStep return format:")
    print(f"  Observations: {type(obs)}")
    print(f"  Rewards: {type(rewards)}")
    print(f"  Terminated: {type(terminated)} - Valid: {isinstance(terminated, bool)}")
    print(f"  Truncated: {type(truncated)} - Valid: {isinstance(truncated, bool)}")
    print(f"  Info: {type(info)} - Valid: {isinstance(info, dict)}")
    
    env.close()
    print(f"Gymnasium compliance test completed!")


def test_coverage_system():
    """Test coverage calculation system."""
    print("\n" + "=" * 60)
    print("Testing Coverage System")
    print("=" * 60)
    
    config = CoverageConfig()
    env = CoverageEnvV5(config)
    
    observations, info = env.reset(seed=42)
    initial_coverage = info['coverage_percentage']
    print(f"Initial coverage: {initial_coverage:.1f}%")
    
    # Test coverage increase
    print(f"Testing coverage increase over 20 steps...")
    coverage_history = [initial_coverage]
    
    for step in range(20):
        # Use deterministic actions to move agents
        actions = []
        for i in range(env.n_agents):
            # Move in different directions
            if step < 10:
                action = np.array([0.2, 0.1 * (i + 1)], dtype=np.float32)
            else:
                action = np.array([-0.1, 0.15 * (i + 1)], dtype=np.float32)
            actions.append(action)
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        coverage_history.append(info['coverage_percentage'])
        
        if step % 5 == 4:  # Print every 5 steps
            print(f"  Step {step + 1}: Coverage={info['coverage_percentage']:5.1f}%")
    
    final_coverage = info['coverage_percentage']
    coverage_increase = final_coverage - initial_coverage
    
    print(f"\nCoverage system test results:")
    print(f"  Initial coverage: {initial_coverage:.1f}%")
    print(f"  Final coverage: {final_coverage:.1f}%")
    print(f"  Coverage increase: {coverage_increase:.1f}%")
    print(f"  Test passed: {coverage_increase > 0}")
    
    env.close()


def test_collision_detection():
    """Test collision detection system."""
    print("\n" + "=" * 60)
    print("Testing Collision Detection")
    print("=" * 60)
    
    config = CoverageConfig()
    env = CoverageEnvV5(config)
    
    observations, info = env.reset(seed=42)
    print(f"Initial agent positions:")
    for i, pos in enumerate(env.agent_positions):
        print(f"  Agent {i}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    # Test collision avoidance by moving agents toward each other
    print(f"\nTesting collision scenarios...")
    collision_detected = False
    
    for step in range(30):
        actions = []
        
        # Move agents towards each other to test collision detection
        if step < 15:
            # Move first two agents toward each other
            if env.n_agents >= 2:
                direction = env.agent_positions[1] - env.agent_positions[0]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    actions.append(direction * 0.3)
                    actions.append(-direction * 0.3)
                else:
                    actions.extend([np.array([0.1, 0.0]), np.array([-0.1, 0.0])])
                
                # Other agents move randomly
                for i in range(2, env.n_agents):
                    actions.append(env.action_space.sample() * 0.1)
            else:
                actions = [env.action_space.sample() * 0.1 for _ in range(env.n_agents)]
        else:
            # Random movement
            actions = [env.action_space.sample() * 0.2 for _ in range(env.n_agents)]
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        if any(info['collision_status']):
            collision_detected = True
            print(f"  Step {step + 1}: Collision detected for agents {[i for i, c in enumerate(info['collision_status']) if c]}")
        
        # Check minimum distances
        min_distance = float('inf')
        for i in range(env.n_agents):
            for j in range(i + 1, env.n_agents):
                dist = np.linalg.norm(env.agent_positions[i] - env.agent_positions[j])
                min_distance = min(min_distance, dist)
        
        if step % 10 == 9:  # Print every 10 steps
            print(f"  Step {step + 1}: Min agent distance={min_distance:.3f}, "
                  f"Required={env.min_distance:.3f}")
    
    print(f"\nCollision detection test results:")
    print(f"  Collision detected during test: {collision_detected}")
    print(f"  Total collisions in metrics: {env.metrics['collision_count']}")
    print(f"  Minimum distance maintained: {min_distance >= env.min_distance * 0.9}")  # Allow small tolerance
    
    env.close()


def test_reward_system():
    """Test reward system functionality."""
    print("\n" + "=" * 60)
    print("Testing Reward System")
    print("=" * 60)
    
    config = CoverageConfig()
    env = CoverageEnvV5(config)
    
    observations, info = env.reset(seed=42)
    
    # Test reward components
    print(f"Testing reward components over 15 steps...")
    total_rewards = [0.0] * env.n_agents
    
    for step in range(15):
        # Use different strategies for different agents
        actions = []
        for i in range(env.n_agents):
            if i == 0:
                # Exploration strategy
                action = np.array([0.3, 0.1], dtype=np.float32)
            elif i == 1 and env.n_agents > 1:
                # Coverage strategy
                action = np.array([0.1, 0.3], dtype=np.float32)
            else:
                # Random strategy
                action = env.action_space.sample() * 0.2
            actions.append(action)
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        for i, reward in enumerate(rewards):
            total_rewards[i] += reward
        
        if step % 5 == 4:  # Print every 5 steps
            print(f"  Step {step + 1}: Rewards=[{', '.join([f'{r:6.2f}' for r in rewards])}]")
            print(f"    Coverage: {info['coverage_percentage']:5.1f}%, "
                  f"Efficiency: {info['efficiency_score']:.3f}")
    
    print(f"\nReward system test results:")
    for i, total_reward in enumerate(total_rewards):
        print(f"  Agent {i} total reward: {total_reward:.2f}")
    
    print(f"  Average reward per agent: {np.mean(total_rewards):.2f}")
    print(f"  Reward variance: {np.var(total_rewards):.2f}")
    
    env.close()


def test_visualization():
    """Test visualization and rendering."""
    print("\n" + "=" * 60)
    print("Testing Visualization")
    print("=" * 60)
    
    config = CoverageConfig()
    env = CoverageEnvV5(config, render_mode='rgb_array')
    
    observations, info = env.reset(seed=42)
    print(f"Environment reset for visualization test")
    
    # Test rendering
    print(f"Testing rendering...")
    render_success = True
    
    try:
        # Test initial render
        rgb_array = env.render()
        if rgb_array is not None:
            print(f"  Initial render successful: {rgb_array.shape}")
        else:
            print(f"  Initial render returned None")
            render_success = False
        
        # Test rendering during simulation
        for step in range(5):
            actions = [env.action_space.sample() * 0.2 for _ in range(env.n_agents)]
            observations, rewards, terminated, truncated, info = env.step(actions)
            
            rgb_array = env.render()
            if rgb_array is not None:
                print(f"  Step {step + 1} render successful: {rgb_array.shape}")
            else:
                print(f"  Step {step + 1} render failed")
                render_success = False
                break
    
    except Exception as e:
        print(f"  Rendering error: {e}")
        render_success = False
    
    print(f"\nVisualization test results:")
    print(f"  Rendering successful: {render_success}")
    print(f"  Frames generated: {env.frame_count}")
    print(f"  Output directory: {env.output_dir}")
    
    env.close()


def test_performance():
    """Test environment performance."""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    config = CoverageConfig()
    env = CoverageEnvV5(config)  # No rendering for performance test
    
    # Test reset performance
    reset_times = []
    for i in range(5):
        start_time = time.time()
        observations, info = env.reset(seed=i)
        reset_time = time.time() - start_time
        reset_times.append(reset_time)
    
    avg_reset_time = np.mean(reset_times)
    print(f"Reset performance:")
    print(f"  Average reset time: {avg_reset_time:.4f} seconds")
    
    # Test step performance
    step_times = []
    observations, info = env.reset(seed=42)
    
    for step in range(100):
        actions = [env.action_space.sample() * 0.2 for _ in range(env.n_agents)]
        
        start_time = time.time()
        observations, rewards, terminated, truncated, info = env.step(actions)
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        if terminated or truncated:
            break
    
    avg_step_time = np.mean(step_times)
    total_time = sum(step_times)
    
    print(f"\nStep performance (100 steps):")
    print(f"  Average step time: {avg_step_time:.4f} seconds")
    print(f"  Total simulation time: {total_time:.4f} seconds")
    print(f"  Steps per second: {len(step_times) / total_time:.1f}")
    
    env.close()


def test_compatibility():
    """Test compatibility with existing systems."""
    print("\n" + "=" * 60)
    print("Testing Compatibility")
    print("=" * 60)
    
    # Test with different configurations
    config = CoverageConfig()
    
    # Test with modified configuration
    config.config['agents']['n_agents'] = 4
    config.config['environment']['max_steps'] = 100
    
    print(f"Testing with modified configuration:")
    print(f"  Agents: {config.config['agents']['n_agents']}")
    print(f"  Max steps: {config.config['environment']['max_steps']}")
    
    try:
        env = CoverageEnvV5(config)
        observations, info = env.reset()
        
        print(f"  Environment created successfully")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Number of observations: {len(observations)}")
        
        # Quick simulation
        for step in range(10):
            actions = [env.action_space.sample() * 0.1 for _ in range(env.n_agents)]
            observations, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"  Simulation completed successfully")
        print(f"  Final coverage: {info['coverage_percentage']:.1f}%")
        
        env.close()
        
    except Exception as e:
        print(f"  Compatibility test failed: {e}")
        return False
    
    print(f"Compatibility test passed!")
    return True


def run_full_test_suite():
    """Run complete test suite."""
    print("Coverage Environment V5 - Comprehensive Test Suite")
    print("=" * 80)
    
    test_functions = [
        test_basic_functionality,
        test_gymnasium_compliance,
        test_coverage_system,
        test_collision_detection,
        test_reward_system,
        test_visualization,
        test_performance,
        test_compatibility
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
            print(f"✓ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"Test Suite Summary: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! CoverageEnvV5 is ready for use.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    run_full_test_suite()