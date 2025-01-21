import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class QNetwork(nn.Module):
    """
    Feed-forward network for estimating Q-values with residual connections.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        """
        Args:
            input_dim (int): Size of the input (state) vector.
            output_dim (int): Number of possible actions (e.g., size of dictionary).
            hidden_dim (int): Number of hidden units (tunable).
        """
        super(QNetwork, self).__init__()
        
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Hidden layers with residual connections
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # Handle single sample case during inference
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Input layer
        x = self.input_layer(x)
        
        # First residual block
        identity = x
        x = self.hidden_layer1(x)
        x = F.relu(x + identity)
        x = self.dropout(x)
        
        # Second residual block
        identity = x
        x = self.hidden_layer2(x)
        x = F.relu(x + identity)
        x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Remove batch dimension if it was added
        if not self.training and x.size(0) == 1:
            x = x.squeeze(0)
            
        return x


class DQNAgent:
    """
    Deep Q-Network Agent:
      - Maintains an online Q-network and a target Q-network.
      - Uses epsilon-greedy exploration to select actions.
      - Periodically updates target network weights.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        target_update_freq=1000,
        device=None
    ):
        """
        Args:
            state_dim (int): Dimension of the flattened state input.
            action_dim (int): Number of possible actions (e.g., # of words in dictionary).
            hidden_dim (int): Hidden layer size for QNetwork.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for Q-learning.
            epsilon_start (float): Initial value of epsilon for exploration.
            epsilon_end (float): Minimum value of epsilon after decay.
            epsilon_decay (float): Multiplicative factor for epsilon decay each step.
            target_update_freq (int): Frequency (in steps) to update target network.
            device (str or torch.device): "cpu" or "cuda" device for PyTorch.
        """
        # Input validation
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1")
        if not (0 <= epsilon_start <= 1) or not (0 <= epsilon_end <= 1):
            raise ValueError("epsilon values must be between 0 and 1")
        if not (0 < epsilon_decay <= 1):
            raise ValueError("epsilon_decay must be between 0 and 1")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Statistics for monitoring
        self.train_steps = 0
        self.updates = 0
        self.avg_loss = 0
        self.avg_q = 0

        # Build networks
        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def select_action(self, state, valid_mask, eval_mode=False):
        """
        Epsilon-greedy action selection.
        Args:
            state (np.ndarray or torch.Tensor): The current state, shape [state_dim].
            valid_mask (np.ndarray or list[bool]): Boolean mask of valid actions (words).
            eval_mode (bool): If True, uses a very small epsilon for evaluation.

        Returns:
            action (int): Index of the chosen action (word) in [0, action_dim).
        """
        # Input validation
        if not isinstance(valid_mask, (np.ndarray, list)):
            raise TypeError("valid_mask must be a numpy array or list")
        if len(valid_mask) != self.action_dim:
            raise ValueError(f"valid_mask length ({len(valid_mask)}) must match action_dim ({self.action_dim})")
            
        # In eval mode, use a very small epsilon
        if not eval_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Convert state to torch.Tensor if not already
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif not isinstance(state, torch.Tensor):
            raise TypeError("state must be either numpy array or torch tensor")
            
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Use a very small epsilon in eval mode
        current_epsilon = 0.01 if eval_mode else self.epsilon

        if random.random() > current_epsilon:
            with torch.no_grad():
                q_values = self.online_net(state)
                
                # Set Q-values of invalid actions to negative infinity
                q_values = q_values.squeeze()
                invalid_mask = ~torch.tensor(valid_mask, device=self.device)
                q_values[invalid_mask] = float('-inf')
                
                # Update statistics
                if not eval_mode:
                    self.avg_q = 0.95 * self.avg_q + 0.05 * q_values.max().item()
                
                return q_values.argmax().item()
        else:
            # Random choice from valid actions
            valid_indices = np.where(valid_mask)[0]
            return random.choice(valid_indices)

    def learn(self, batch):
        """
        Update the Q-network using a batch of transitions.
        Implements Double DQN for more stable learning.
        """
        states = torch.tensor(batch['states'], device=self.device)
        actions = torch.tensor(batch['actions'], device=self.device)
        rewards = torch.tensor(batch['rewards'], device=self.device)
        next_states = torch.tensor(batch['next_states'], device=self.device)
        dones = torch.tensor(batch['dones'], device=self.device)

        # Compute current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use online network to select actions, target network to evaluate them
        with torch.no_grad():
            # Select actions using online network
            online_next_q_values = self.online_net(next_states)
            best_actions = online_next_q_values.argmax(dim=1, keepdim=True)
            
            # Evaluate Q-values using target network
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.gather(1, best_actions)
            
            # Compute target Q values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * max_next_q_values

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)  # Huber loss for stability
        
        # Update statistics
        self.avg_loss = 0.95 * self.avg_loss + 0.05 * loss.item()
        self.train_steps += 1
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update target network if it's time
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.updates += 1

        return loss.item()
    
    def get_statistics(self):
        """Return current training statistics."""
        return {
            'train_steps': self.train_steps,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'avg_loss': self.avg_loss,
            'avg_q': self.avg_q
        }
        
    def save(self, path):
        """Save the online network state dict."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'epsilon': self.epsilon,
            'statistics': self.get_statistics()
        }, path)
        
    def load(self, path):
        """Load a saved state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['online_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_steps = checkpoint['train_steps']
        self.epsilon = checkpoint['epsilon']
        
        # Restore statistics if available
        if 'statistics' in checkpoint:
            stats = checkpoint['statistics']
            self.updates = stats.get('updates', 0)
            self.avg_loss = stats.get('avg_loss', 0)
            self.avg_q = stats.get('avg_q', 0)