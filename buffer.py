import numpy as np


class MultiAgentReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        """
        Initialize the replay buffer

        Args:
            max_size: Maximum number of transitions to store
            critic_dims: Dimension of global state for critic
            actor_dims: List of dimensions of local observations for each actor
            n_actions: Dimension of action space for each agent
            n_agents: Number of agents
            batch_size: Size of batches to sample
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Memory for critic's global state
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))

        # Memory for rewards and terminal flags
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        # Memory for actors' local observations and actions
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, actor_dims[i]))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, actor_dims[i]))
            )
            self.actor_action_memory.append(
                np.zeros((self.mem_size, n_actions))
            )

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        """
        Store a transition in the buffer

        Args:
            raw_obs: List of local observations for each agent
            state: Global state for critic
            action: List of actions taken by each agent
            reward: List of rewards received by each agent
            raw_obs_: List of next observations for each agent
            state_: Next global state for critic
            done: List of terminal flags for each agent
        """
        # Get the next index in circular buffer
        index = self.mem_cntr % self.mem_size

        # Store observations and actions for each agent
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        # Store global state, reward, and terminal flags
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # Increment counter
        self.mem_cntr += 1

    def sample_buffer(self):
        """
        Sample a batch of transitions from the buffer

        Returns:
            actor_states: List of state batches for each actor
            states: Batch of global states for critic
            actions: List of action batches for each agent
            rewards: Batch of rewards for each agent
            actor_new_states: List of next state batches for each actor
            states_: Batch of next global states for critic
            terminal: Batch of terminal flags for each agent
        """
        # Maximum valid index
        max_mem = min(self.mem_cntr, self.mem_size)

        # Sample batch of indices
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Get states, rewards, next states, and terminal flags
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # Get observations and actions for each agent
        actor_states = []
        actor_new_states = []
        actions = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        """Check if enough transitions are stored for sampling a batch"""
        return self.mem_cntr >= self.batch_size

    def clear(self):
        """Clear the replay buffer"""
        self.mem_cntr = 0