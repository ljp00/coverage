"""
Simple evaluation script for coverage model - Fixed version
Easy checkpoint path configuration - just change the folder name
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import json
import time
import argparse
from coverage_env_v4 import CoverageEnv
from config_v2 import CoverageConfig
from maddpg_coverage import MADDPGCoverage
from visualization import CoverageVisualizer


# Configuration - only modify these values
BASE_CHECKPOINT_PATH = r"D:\python_project\coverage0714\checkpoints"
CHECKPOINT_FOLDER_NAME = "maddpg_coverage_3agents_20250731-140309"
MODEL_TYPE = "epbest"  # Changed from "best" to "epbest"

# Evaluation settings
NUM_EPISODES = 5
RENDER_EPISODES = True
SAVE_RESULTS = True
VERBOSE_OUTPUT = True

# Auto-build full path
FULL_CHECKPOINT_PATH = os.path.join(BASE_CHECKPOINT_PATH, CHECKPOINT_FOLDER_NAME)


def check_checkpoint_availability(checkpoint_dir: str, model_type: str = "epbest"):
    """Check checkpoint file availability"""

    print(f"Checking checkpoint directory: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Directory not found - {checkpoint_dir}")
        return None

    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    if not pth_files:
        print(f"Error: No .pth files found in directory - {checkpoint_dir}")
        return None

    print(f"Found {len(pth_files)} .pth files")

    # Look for specified model type checkpoint
    required_files = [
        f"actor_0_{model_type}.pth",
        f"critic_0_{model_type}.pth",
        f"target_actor_0_{model_type}.pth",
        f"target_critic_0_{model_type}.pth"
    ]

    existing_files = []
    missing_files = []

    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file)
        else:
            missing_files.append(file)

    if len(existing_files) == 4:
        print(f"Found complete {model_type} model file set")
        return model_type
    else:
        print(f"{model_type} model files incomplete, missing: {missing_files}")

        # Try to find other available models
        available_models = set()
        for file in pth_files:
            if file.startswith('actor_0_') and file.endswith('.pth'):
                model_name = file.replace('actor_0_', '').replace('.pth', '')
                available_models.add(model_name)

        print(f"Available model types: {sorted(list(available_models))}")

        # Priority order for model selection
        priority_models = ['epbest', 'epfinal'] + [f'ep{i}00' for i in range(20, 0, -1)]

        # Try to use priority models first
        for priority_model in priority_models:
            if priority_model in available_models:
                test_files = [
                    f"actor_0_{priority_model}.pth",
                    f"critic_0_{priority_model}.pth",
                    f"target_actor_0_{priority_model}.pth",
                    f"target_critic_0_{priority_model}.pth"
                ]

                if all(os.path.exists(os.path.join(checkpoint_dir, f)) for f in test_files):
                    print(f"Using priority model: {priority_model}")
                    return priority_model

        # If no priority model found, use first available complete model
        for available_model in sorted(available_models, reverse=True):
            test_files = [
                f"actor_0_{available_model}.pth",
                f"critic_0_{available_model}.pth",
                f"target_actor_0_{available_model}.pth",
                f"target_critic_0_{available_model}.pth"
            ]

            if all(os.path.exists(os.path.join(checkpoint_dir, f)) for f in test_files):
                print(f"Using available model: {available_model}")
                return available_model

        print("No complete model file set found")
        return None


class SafeCoverageEnvV2:
    """Wrapper for CoverageEnvV2 with error handling"""

    def __init__(self, config):
        self.env = CoverageEnv(config)

        # Copy all attributes
        for attr in dir(self.env):
            if not attr.startswith('_') and not callable(getattr(self.env, attr)):
                setattr(self, attr, getattr(self.env, attr))

    def reset(self):
        """Safe reset with error handling"""
        try:
            return self.env.reset()
        except Exception as e:
            print(f"Warning: Error in reset: {e}")
            # Try to initialize manually if needed
            return self.env.reset()

    def step(self, actions):
        """Safe step with error handling"""
        try:
            return self.env.step(actions)
        except Exception as e:
            if "total_cells" in str(e):
                print(f"Warning: Exploration reward error, continuing without it")
                # Continue without exploration rewards
                try:
                    # Get basic step result
                    observations, rewards, terminated, truncated, info = self.env._basic_step(actions)
                    return observations, rewards, terminated, truncated, info
                except:
                    # Fallback: return current state
                    return self.env._get_observations(), [0.0] * self.env.n_agents, False, False, {'coverage': 0.0}
            else:
                raise e

    def close(self):
        """Safe close"""
        try:
            self.env.close()
        except:
            pass

    def render(self):
        """Safe render"""
        try:
            return self.env.render()
        except Exception as e:
            print(f"Warning: Render error: {e}")


def run_evaluation():
    """Run model evaluation"""

    print("=" * 70)
    print("Multi-Agent Coverage Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint path: {FULL_CHECKPOINT_PATH}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Evaluation episodes: {NUM_EPISODES}")
    print("=" * 70)

    # Check checkpoint availability
    available_model = check_checkpoint_availability(FULL_CHECKPOINT_PATH, MODEL_TYPE)

    if available_model is None:
        print("Evaluation failed: No available checkpoint files found")
        return None

    # Load configuration and environment
    print("\nInitializing environment...")
    config = CoverageConfig()
    env = SafeCoverageEnvV2(config)  # Use safe wrapper

    print(f"Environment configuration:")
    print(f"  - Number of agents: {env.n_agents}")
    print(f"  - Observation dimension: {env.observation_space.shape}")
    print(f"  - Action dimension: {env.action_space.shape}")
    print(f"  - World size: {env.world_size}")
    print(f"  - Max steps: {env.max_steps}")

    # Setup MADDPG
    obs_dims = [env.observation_space.shape[0]] * env.n_agents
    action_dim = env.action_space.shape[0]

    maddpg = MADDPGCoverage(
        n_agents=env.n_agents,
        obs_dims=obs_dims,
        action_dims=action_dim
    )

    # Load trained model
    print(f"\nLoading model...")
    maddpg.chkpt_dir = FULL_CHECKPOINT_PATH

    try:
        maddpg.load_checkpoint(episode=available_model)
        print(f"Model loaded successfully: {available_model}")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

    # Run evaluation
    print(f"\nStarting evaluation ({NUM_EPISODES} episodes)...")
    print("-" * 50)

    results = {
        'coverage': [],
        'rewards': [],
        'steps': [],
        'safety_violations': []
    }

    for episode in range(NUM_EPISODES):
        if VERBOSE_OUTPUT:
            print(f"Running episode {episode + 1}/{NUM_EPISODES}")

        try:
            # Reset environment
            observations, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            safety_violations = 0
            coverage_history = [info.get('coverage', 0.0)]

            done = False
            max_steps_reached = False

            while not done and episode_steps < env.max_steps and not max_steps_reached:
                # Render if needed
                if RENDER_EPISODES:
                    env.render()

                # Choose actions (evaluation mode, no exploration)
                actions = maddpg.choose_actions(observations, evaluate=True)

                # Environment step
                try:
                    observations, rewards, terminated, truncated, info = env.step(actions)

                    # Record data
                    coverage_history.append(info.get('coverage', coverage_history[-1]))

                    # Count safety violations
                    if 'safety_assessment' in info:
                        for agent_assessment in info['safety_assessment'].values():
                            safety_violations += len(agent_assessment.get('violations', []))

                    episode_reward += np.mean(rewards) if isinstance(rewards, (list, np.ndarray)) else rewards
                    episode_steps += 1
                    done = terminated or truncated

                except Exception as step_error:
                    print(f"Warning: Step error in episode {episode + 1}, step {episode_steps}: {step_error}")
                    # Try to continue or break if too many errors
                    episode_steps += 1
                    if episode_steps >= env.max_steps:
                        max_steps_reached = True
                        break

            # Save episode results
            final_coverage = coverage_history[-1] if coverage_history else 0.0
            results['coverage'].append(final_coverage)
            results['rewards'].append(episode_reward)
            results['steps'].append(episode_steps)
            results['safety_violations'].append(safety_violations)

            if VERBOSE_OUTPUT:
                print(f"  Episode {episode + 1}: Coverage={final_coverage:.1f}%, "
                      f"Reward={episode_reward:.2f}, Steps={episode_steps}, "
                      f"Safety violations={safety_violations}")

        except Exception as episode_error:
            print(f"Error in episode {episode + 1}: {episode_error}")
            # Add default values for failed episode
            results['coverage'].append(0.0)
            results['rewards'].append(0.0)
            results['steps'].append(0)
            results['safety_violations'].append(0)

    env.close()

    # Calculate statistics
    if not results['coverage']:
        print("No successful episodes to analyze")
        return None

    avg_coverage = np.mean(results['coverage'])
    std_coverage = np.std(results['coverage'])
    avg_reward = np.mean(results['rewards'])
    avg_steps = np.mean(results['steps'])
    avg_violations = np.mean(results['safety_violations'])

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Average Coverage: {avg_coverage:.1f}% ± {std_coverage:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Safety Violations: {avg_violations:.1f}")
    print(f"Best Coverage: {max(results['coverage']):.1f}%")
    print(f"Worst Coverage: {min(results['coverage']):.1f}%")
    print(f"Coverage Standard Deviation: {std_coverage:.1f}%")

    # Performance level assessment
    if avg_coverage >= 90:
        performance_level = "Excellent"
    elif avg_coverage >= 80:
        performance_level = "Good"
    elif avg_coverage >= 70:
        performance_level = "Acceptable"
    else:
        performance_level = "Needs Improvement"

    print(f"Performance Level: {performance_level}")
    print("=" * 70)

    # Save results if needed
    if SAVE_RESULTS:
        results_dir = f"evaluation_results_{int(time.time())}"
        os.makedirs(results_dir, exist_ok=True)

        save_data = {
            'checkpoint_info': {
                'path': FULL_CHECKPOINT_PATH,
                'model_type': available_model,
                'folder_name': CHECKPOINT_FOLDER_NAME
            },
            'evaluation_settings': {
                'num_episodes': NUM_EPISODES,
                'render_episodes': RENDER_EPISODES,
                'verbose_output': VERBOSE_OUTPUT
            },
            'statistics': {
                'avg_coverage': float(avg_coverage),
                'std_coverage': float(std_coverage),
                'avg_reward': float(avg_reward),
                'avg_steps': float(avg_steps),
                'avg_violations': float(avg_violations),
                'best_coverage': float(max(results['coverage'])),
                'worst_coverage': float(min(results['coverage'])),
                'performance_level': performance_level
            },
            'detailed_results': {
                'coverage': [float(c) for c in results['coverage']],
                'rewards': [float(r) for r in results['rewards']],
                'steps': [int(s) for s in results['steps']],
                'safety_violations': [int(v) for v in results['safety_violations']]
            }
        }

        json_file = os.path.join(results_dir, 'evaluation_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {results_dir}")
        print(f"JSON file: {json_file}")

    return results


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Simple coverage model evaluation script')
    parser.add_argument('--folder', type=str, default=None,
                        help='Checkpoint folder name (overrides default configuration)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of evaluation episodes (overrides default configuration)')
    parser.add_argument('--render', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--model', type=str, default=None,
                        help='Model type to load (e.g., epbest, epfinal, ep1900)')

    args = parser.parse_args()

    # Apply command line arguments
    global CHECKPOINT_FOLDER_NAME, NUM_EPISODES, RENDER_EPISODES, FULL_CHECKPOINT_PATH, MODEL_TYPE

    if args.folder:
        CHECKPOINT_FOLDER_NAME = args.folder
        FULL_CHECKPOINT_PATH = os.path.join(BASE_CHECKPOINT_PATH, CHECKPOINT_FOLDER_NAME)
        print(f"Using command line specified folder: {CHECKPOINT_FOLDER_NAME}")

    if args.episodes:
        NUM_EPISODES = args.episodes
        print(f"Using command line specified episodes: {NUM_EPISODES}")

    if args.render:
        RENDER_EPISODES = True
        print("Visualization rendering enabled")

    if args.model:
        MODEL_TYPE = args.model
        print(f"Using command line specified model type: {MODEL_TYPE}")

    # Run evaluation
    try:
        results = run_evaluation()

        if results is not None:
            print("\nEvaluation completed successfully!")
            return 0
        else:
            print("\nEvaluation failed!")
            return 1

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nEvaluation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())