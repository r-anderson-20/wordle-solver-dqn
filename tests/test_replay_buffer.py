"""
Tests for the replay buffer implementation.
"""

import pytest
import numpy as np
from replay_buffer import ReplayBuffer

class TestReplayBuffer:
    """Test suite for ReplayBuffer class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        buffer = ReplayBuffer(max_size=100, state_dim=10)
        assert buffer.max_size == 100
        assert buffer.state_dim == 10
        assert buffer.size == 0
        assert buffer.ptr == 0
        
    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=0, state_dim=10)
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=-1, state_dim=10)
        with pytest.raises(ValueError, match="state_dim must be positive"):
            ReplayBuffer(max_size=100, state_dim=0)
        with pytest.raises(ValueError, match="state_dim must be positive"):
            ReplayBuffer(max_size=100, state_dim=-1)
            
    def test_add_valid(self):
        """Test adding valid transitions."""
        buffer = ReplayBuffer(max_size=2, state_dim=3)
        
        state = np.array([1, 2, 3], dtype=np.float32)
        next_state = np.array([4, 5, 6], dtype=np.float32)
        
        # Add first transition
        buffer.add(state, 0, 1.0, next_state, False)
        assert buffer.size == 1
        assert buffer.ptr == 1
        np.testing.assert_array_equal(buffer.states[0], state)
        np.testing.assert_array_equal(buffer.next_states[0], next_state)
        assert buffer.actions[0] == 0
        assert buffer.rewards[0] == 1.0
        assert buffer.dones[0] == 0.0
        
        # Add second transition
        buffer.add(next_state, 1, -1.0, state, True)
        assert buffer.size == 2
        assert buffer.ptr == 0  # Wrapped around
        np.testing.assert_array_equal(buffer.states[1], next_state)
        np.testing.assert_array_equal(buffer.next_states[1], state)
        assert buffer.actions[1] == 1
        assert buffer.rewards[1] == -1.0
        assert buffer.dones[1] == 1.0
        
    def test_add_invalid(self):
        """Test adding invalid transitions."""
        buffer = ReplayBuffer(max_size=100, state_dim=2)
        valid_state = np.array([1, 2], dtype=np.float32)
        
        # Invalid state shape
        with pytest.raises(ValueError, match="state must be a numpy array of shape"):
            buffer.add(np.array([1]), 0, 1.0, valid_state, False)
            
        # Invalid next_state shape
        with pytest.raises(ValueError, match="next_state must be a numpy array of shape"):
            buffer.add(valid_state, 0, 1.0, np.array([1]), False)
            
        # Invalid action type
        with pytest.raises(ValueError, match="action must be an integer"):
            buffer.add(valid_state, 0.5, 1.0, valid_state, False)
            
        # Invalid reward type
        with pytest.raises(ValueError, match="reward must be a float"):
            buffer.add(valid_state, 0, "1.0", valid_state, False)
            
        # Invalid done type
        with pytest.raises(ValueError, match="done must be a boolean"):
            buffer.add(valid_state, 0, 1.0, valid_state, 1)
            
    def test_sample_valid(self):
        """Test sampling with valid parameters."""
        buffer = ReplayBuffer(max_size=3, state_dim=2)
        
        # Add some transitions
        states = [
            np.array([1, 2], dtype=np.float32),
            np.array([3, 4], dtype=np.float32),
            np.array([5, 6], dtype=np.float32)
        ]
        
        for i, state in enumerate(states):
            next_state = states[(i + 1) % len(states)]
            buffer.add(state, i, float(i), next_state, i == len(states) - 1)
            
        # Sample single transition
        batch = buffer.sample(1)
        assert isinstance(batch, dict)
        assert all(k in batch for k in ['states', 'actions', 'rewards', 'next_states', 'dones'])
        assert all(isinstance(v, np.ndarray) for v in batch.values())
        assert all(len(v) == 1 for v in batch.values())
        
        # Sample multiple transitions
        batch = buffer.sample(2)
        assert all(len(v) == 2 for v in batch.values())
        
        # Verify shapes
        assert batch['states'].shape == (2, 2)
        assert batch['actions'].shape == (2,)
        assert batch['rewards'].shape == (2,)
        assert batch['next_states'].shape == (2, 2)
        assert batch['dones'].shape == (2,)
        
    def test_sample_invalid(self):
        """Test sampling with invalid parameters."""
        buffer = ReplayBuffer(max_size=100, state_dim=2)
        
        # Empty buffer
        with pytest.raises(ValueError, match="Cannot sample from an empty buffer"):
            buffer.sample(1)
            
        # Add one transition
        state = np.array([1, 2], dtype=np.float32)
        buffer.add(state, 0, 1.0, state, False)
        
        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            buffer.sample(0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            buffer.sample(-1)
            
        # Batch size too large
        with pytest.raises(ValueError, match="Cannot sample .* transitions"):
            buffer.sample(2)
            
    def test_buffer_overflow(self):
        """Test FIFO behavior when buffer is full."""
        buffer = ReplayBuffer(max_size=2, state_dim=2)
        
        states = [
            np.array([1, 2], dtype=np.float32),
            np.array([3, 4], dtype=np.float32),
            np.array([5, 6], dtype=np.float32)
        ]
        
        # Fill buffer
        buffer.add(states[0], 0, 0.0, states[1], False)  # Goes to index 0
        buffer.add(states[1], 1, 1.0, states[2], False)  # Goes to index 1
        assert buffer.size == 2
        assert buffer.ptr == 0  # Points to where next item will go
        
        # Add one more transition (should replace oldest at index 0)
        buffer.add(states[2], 2, 2.0, states[0], True)
        assert buffer.size == 2
        assert buffer.ptr == 1  # Points to next position
        
        # Verify the oldest transition was replaced
        # The new transition should be at index 0
        np.testing.assert_array_equal(buffer.states[0], states[2])
        assert buffer.actions[0] == 2
        assert buffer.rewards[0] == 2.0
        assert buffer.dones[0] == 1.0
        
        # The second transition should still be at index 1
        np.testing.assert_array_equal(buffer.states[1], states[1])
        assert buffer.actions[1] == 1
        assert buffer.rewards[1] == 1.0
        assert buffer.dones[1] == 0.0
