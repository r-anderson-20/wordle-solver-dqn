"""
Experience replay buffer implementation for DQN training.
Provides efficient storage and sampling of transitions (state, action, reward, next_state, done)
using numpy arrays. Implements FIFO behavior when buffer is full.
"""

import numpy as np

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for Q-learning.
    Stores transitions using numpy arrays for efficient sampling and memory usage.
    When buffer is full, oldest transitions are replaced first.
    """

    def __init__(self, max_size, state_dim):
        """
        Initialize replay buffer with fixed maximum size.

        Args:
            max_size (int): Maximum number of transitions to store
            state_dim (int): Dimension of the flattened state vector
                           (for example, 390 + dictionary_size + 1)

        Raises:
            ValueError: If max_size or state_dim are not positive
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
            
        self.max_size = max_size
        self.state_dim = state_dim

        # Create numpy arrays to hold each component of the transition
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

        self.ptr = 0        # Current insert pointer
        self.size = 0       # Current number of stored transitions

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): Current state vector [state_dim]
            action (int): Action index taken
            reward (float): Reward received
            next_state (np.ndarray): Next state vector [state_dim]
            done (bool): Whether episode ended after this transition
            
        Raises:
            ValueError: If inputs have incorrect shapes or types
            
        Notes:
            When buffer is full, oldest transitions are overwritten first
        """
        # Input validation
        if not isinstance(state, np.ndarray) or state.shape != (self.state_dim,):
            raise ValueError(f"state must be a numpy array of shape ({self.state_dim},)")
        if not isinstance(next_state, np.ndarray) or next_state.shape != (self.state_dim,):
            raise ValueError(f"next_state must be a numpy array of shape ({self.state_dim},)")
        if not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not isinstance(reward, (float, np.floating)):
            raise ValueError("reward must be a float")
        if not isinstance(done, bool):
            raise ValueError("done must be a boolean")

        idx = self.ptr

        # Convert inputs to correct types if needed
        self.states[idx] = state.astype(np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = next_state.astype(np.float32)
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) where:
                - states: np.ndarray [batch_size, state_dim]
                - actions: np.ndarray [batch_size]
                - rewards: np.ndarray [batch_size]
                - next_states: np.ndarray [batch_size, state_dim]
                - dones: np.ndarray [batch_size]
                
        Raises:
            ValueError: If batch_size > current buffer size
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} transitions, buffer only has {self.size}")

        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
        }
        return batch
        
    def __len__(self):
        """Returns the current size of the buffer."""
        return self.size