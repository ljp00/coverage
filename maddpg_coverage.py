import torch as T
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork
from buffer import MultiAgentReplayBuffer
import os
import time


class MADDPGCoverage:
    """
    Multi-Agent Deep Deterministic Policy Gradient implementation
    optimized for the coverage task.
    """

    def __init__(self, n_agents, obs_dims, action_dims,
                 lr_actor=0.0001, lr_critic=0.0002, fc1=256, fc2=256,
                 gamma=0.99, tau=0.005, batch_size=256, memory_size=1000000,
                 warmup_steps=1000, noise_scale=0.2, noise_decay=0.9999):
        """
        Initialize the MADDPG algorithm

        Args:
            n_agents: Number of agents
            obs_dims: List of observation dimensions for each agent
            action_dims: Action dimension (same for all agents)
            lr_actor: Learning rate for actor networks
            lr_critic: Learning rate for critic networks
            fc1: Size of first hidden layer
            fc2: Size of second hidden layer
            gamma: Discount factor
            tau: Target network update rate
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            warmup_steps: Number of steps before training begins
            noise_scale: Initial exploration noise scale
            noise_decay: Exploration noise decay factor
        """
        self.agents = []
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.min_noise = 0.01

        # Ensure directory exists
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.chkpt_dir = f'checkpoints/maddpg_coverage_{n_agents}agents_{timestamp}'
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create networks for each agent
        for i in range(n_agents):
            agent = {
                'actor': ActorNetwork(
                    alpha=lr_actor,
                    input_dims=obs_dims[i],
                    fc1_dims=fc1,
                    fc2_dims=fc2,
                    n_actions=action_dims,
                    name=f'actor_{i}',
                    chkpt_dir=self.chkpt_dir
                ),
                'critic': CriticNetwork(
                    beta=lr_critic,
                    input_dims=sum(obs_dims),
                    fc1_dims=fc1,
                    fc2_dims=fc2,
                    n_agents=n_agents,
                    n_actions=action_dims,
                    name=f'critic_{i}',
                    chkpt_dir=self.chkpt_dir
                ),
                'target_actor': ActorNetwork(
                    alpha=lr_actor,
                    input_dims=obs_dims[i],
                    fc1_dims=fc1,
                    fc2_dims=fc2,
                    n_actions=action_dims,
                    name=f'target_actor_{i}',
                    chkpt_dir=self.chkpt_dir
                ),
                'target_critic': CriticNetwork(
                    beta=lr_critic,
                    input_dims=sum(obs_dims),
                    fc1_dims=fc1,
                    fc2_dims=fc2,
                    n_agents=n_agents,
                    n_actions=action_dims,
                    name=f'target_critic_{i}',
                    chkpt_dir=self.chkpt_dir
                )
            }

            # Initialize target networks
            self._update_target_networks(agent, tau=1.0)

            self.agents.append(agent)

        # Initialize experience replay buffer
        self.memory = MultiAgentReplayBuffer(
            max_size=memory_size,
            critic_dims=sum(obs_dims),
            actor_dims=obs_dims,
            n_actions=action_dims,
            n_agents=n_agents,
            batch_size=batch_size
        )

        # Track training metrics
        self.actor_losses = [[] for _ in range(n_agents)]
        self.critic_losses = [[] for _ in range(n_agents)]
        self.avg_q_values = [[] for _ in range(n_agents)]

    def choose_actions(self, observations, evaluate=False):
        """
        Select actions for each agent based on their observations

        Args:
            observations: List of observations for each agent
            evaluate: If True, no exploration noise is added
        """
        self.total_steps += 1
        actions = []

        # Current noise scale with decay
        current_noise_scale = max(
            self.min_noise,
            self.noise_scale * (self.noise_decay ** (self.total_steps // 1000))
        )

        for agent_idx, agent in enumerate(self.agents):
            obs = T.tensor(observations[agent_idx], dtype=T.float32).to(self.device)
            obs = obs.unsqueeze(0)  # Add batch dimension

            with T.no_grad():
                # Get action from actor network
                mu = agent['actor'](obs)

                if not evaluate:
                    # Add exploration noise
                    noise = T.randn_like(mu).to(self.device) * current_noise_scale
                    action = mu + noise
                else:
                    action = mu

                # Clip action to valid range
                action = T.clamp(action, -0.5, 0.5)

            actions.append(action.cpu().detach().numpy()[0])

        return actions

    def store_transition(self, observations, state, actions, rewards, next_observations, next_state, dones):
        """Store experience in replay buffer"""
        self.memory.store_transition(observations, state, actions, rewards, next_observations, next_state, dones)

    def learn(self):
        """Update policy and value networks"""
        # Skip learning if buffer is not full enough or during warmup
        if not self.memory.ready() or self.total_steps < self.warmup_steps:
            return

        # Sample experience from replay buffer
        actor_states, states, actions, rewards, actor_new_states, states_, dones = \
            self.memory.sample_buffer()

        # Process data
        states = T.tensor(states, dtype=T.float32).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        states_ = T.tensor(states_, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        # Process states for each agent
        for agent_idx, agent in enumerate(self.agents):
            # Convert numpy states to tensors
            actor_state = T.tensor(actor_states[agent_idx],
                                   dtype=T.float32).to(self.device)
            actor_new_state = T.tensor(actor_new_states[agent_idx],
                                       dtype=T.float32).to(self.device)

            # Get actions for next state
            new_pi = agent['target_actor'](actor_new_state)
            all_agents_new_actions.append(new_pi)

            # Get actions for current state
            mu = agent['actor'](actor_state)
            all_agents_new_mu_actions.append(mu)

            # Convert old actions to tensor
            old_action = T.tensor(actions[agent_idx],
                                  dtype=T.float32).to(self.device)
            old_agents_actions.append(old_action)

        # Concatenate actions for critic input
        new_actions = T.cat(all_agents_new_actions, dim=1)
        mu = T.cat(all_agents_new_mu_actions, dim=1)
        old_actions = T.cat(old_agents_actions, dim=1)

        # Update each agent
        for agent_idx, agent in enumerate(self.agents):
            # ----------- Update critic ----------- #

            # Target Q-value with next state and target actor's policy
            with T.no_grad():
                target_critic_value = agent['target_critic'](states_, new_actions).flatten()
                critic_target = rewards[:, agent_idx] + \
                                self.gamma * target_critic_value * \
                                (1 - dones[:, agent_idx])

            # Current Q-value estimate
            critic_value = agent['critic'](states, old_actions).flatten()

            # Compute critic loss and update
            critic_loss = F.mse_loss(critic_value, critic_target.detach())

            agent['critic'].optimizer.zero_grad()
            critic_loss.backward()
            agent['critic'].optimizer.step()

            # ----------- Update actor ----------- #

            # Replace this agent's action in the joint action
            actions_for_grad = []
            for a_idx in range(self.n_agents):
                if a_idx == agent_idx:
                    actions_for_grad.append(all_agents_new_mu_actions[a_idx])
                else:
                    actions_for_grad.append(all_agents_new_mu_actions[a_idx].detach())

            actions_for_grad = T.cat(actions_for_grad, dim=1)

            # Compute actor loss and update
            actor_loss = -T.mean(agent['critic'](states, actions_for_grad))

            agent['actor'].optimizer.zero_grad()
            actor_loss.backward()
            agent['actor'].optimizer.step()

            # Store losses for tracking
            self.actor_losses[agent_idx].append(actor_loss.item())
            self.critic_losses[agent_idx].append(critic_loss.item())
            self.avg_q_values[agent_idx].append(critic_value.mean().item())

        # Soft-update target networks
        self._update_targets()

    def _update_targets(self):
        """Update all agents' target networks"""
        for agent in self.agents:
            self._update_target_networks(agent, self.tau)

    def _update_target_networks(self, agent, tau):
        """
        Soft update target network parameters:
        θ_target = τ*θ_current + (1-τ)*θ_target
        """
        for target_param, param in zip(agent['target_actor'].parameters(),
                                       agent['actor'].parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

        for target_param, param in zip(agent['target_critic'].parameters(),
                                       agent['critic'].parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def save_checkpoint(self, episode=None):
        """Save model checkpoints"""
        suffix = f"_ep{episode}" if episode is not None else ""

        print(f'Saving checkpoints{suffix}...')
        try:
            for agent_idx, agent in enumerate(self.agents):
                for network_name in ['actor', 'critic', 'target_actor', 'target_critic']:
                    agent[network_name].save_checkpoint(suffix=suffix)
            print('Successfully saved all checkpoints.')
        except Exception as e:
            print(f'Error saving checkpoints: {str(e)}')

    def load_checkpoint(self, episode=None):
        """Load model checkpoints"""
        suffix = f"_ep{episode}" if episode is not None else ""

        print(f'Loading checkpoints{suffix}...')
        try:
            for agent_idx, agent in enumerate(self.agents):
                for network_name in ['actor', 'critic', 'target_actor', 'target_critic']:
                    agent[network_name].load_checkpoint(suffix=suffix)
            print('Successfully loaded all checkpoints.')
        except Exception as e:
            print(f'Error loading checkpoints: {str(e)}')

    def get_training_metrics(self):
        """Return training metrics for visualization"""
        metrics = {
            'actor_losses': [np.mean(losses[-100:]) if losses else 0 for losses in self.actor_losses],
            'critic_losses': [np.mean(losses[-100:]) if losses else 0 for losses in self.critic_losses],
            'avg_q_values': [np.mean(q_vals[-100:]) if q_vals else 0 for q_vals in self.avg_q_values],
            'noise_scale': max(self.min_noise, self.noise_scale * (self.noise_decay ** (self.total_steps // 1000)))
        }
        return metrics