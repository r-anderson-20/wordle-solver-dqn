import numpy as np

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for Q-learning.
    Stores tuples of (state, action, reward, next_state, done).
    """

    def __init__(self, max_size, state_dim):
        """
        Args:
            max_size (int): Maximum number of transitions to store.
            state_dim (int): Dimension of the flattened state vector
                             (for example, 390 + dictionary_size + 1).
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
        Adds a transition to the buffer.
        
        Args:
            state (np.ndarray): Flattened state vector of shape [state_dim].
            action (int): Index of the chosen action.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Flattened next-state vector [state_dim].
            done (bool): Whether the episode ended after this transition.
            
        Raises:
            ValueError: If inputs have incorrect shapes or types.
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
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            batch (dict): A dictionary containing:
                - 'states': shape [batch_size, state_dim]
                - 'actions': shape [batch_size]
                - 'rewards': shape [batch_size]
                - 'next_states': shape [batch_size, state_dim]
                - 'dones': shape [batch_size]
                
        Raises:
            ValueError: If batch_size is invalid or buffer is empty.
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