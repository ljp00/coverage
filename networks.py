import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    """Actor network for MADDPG - policy function μ(s)"""

    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir):
        """
        Initialize the actor network

        Args:
            alpha: Learning rate
            input_dims: Observation space dimension
            fc1_dims: First hidden layer size
            fc2_dims: Second hidden layer size
            n_actions: Action space dimension
            name: Network name for checkpoint
            chkpt_dir: Directory to save checkpoints
        """
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        # Network architecture
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        # Initialize weights for better convergence
        self._init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        """Initialize network weights"""
        # Use Xavier initialization for better training dynamics
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize final layer with smaller weights to limit initial actions
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass through the network

        Args:
            state: Input state tensor

        Returns:
            Action values scaled to proper range
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))

        # Output layer uses tanh to bound values between -1 and 1
        # Then scale to the desired action range (-0.5 to 0.5)
        actions = 0.5 * T.tanh(self.fc3(x))

        return actions

    def save_checkpoint(self, suffix=''):
        """Save model checkpoint"""
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        filepath = f"{self.checkpoint_file}{suffix}.pth"
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load_checkpoint(self, suffix=''):
        """Load model checkpoint"""
        filepath = f"{self.checkpoint_file}{suffix}.pth"

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = T.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class CriticNetwork(nn.Module):
    """Critic network for MADDPG - Q value function Q(s,a)"""

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions,
                 name, chkpt_dir):
        """
        Initialize the critic network

        Args:
            beta: Learning rate
            input_dims: State space dimension (combined for all agents)
            fc1_dims: First hidden layer size
            fc2_dims: Second hidden layer size
            n_agents: Number of agents
            n_actions: Action space dimension per agent
            name: Network name for checkpoint
            chkpt_dir: Directory to save checkpoints
        """
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        # Calculate total action dimension for all agents
        total_actions_dim = n_agents * n_actions

        # Network architecture
        self.fc1 = nn.Linear(input_dims + total_actions_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # Layer normalization
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        # Initialize weights
        self._init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.q.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forward pass through the network

        Args:
            state: State tensor (combined for all agents)
            action: Action tensor (combined for all agents)

        Returns:
            Q-value estimate
        """
        # Concatenate state and action as input
        x = T.cat([state, action], dim=1)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q = self.q(x)

        return q

    def save_checkpoint(self, suffix=''):
        """Save model checkpoint"""
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        filepath = f"{self.checkpoint_file}{suffix}.pth"
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def load_checkpoint(self, suffix=''):
        """Load model checkpoint"""
        filepath = f"{self.checkpoint_file}{suffix}.pth"

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = T.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])