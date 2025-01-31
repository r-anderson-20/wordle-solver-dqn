"""
Tests for the DQN agent implementation.
"""

import pytest
import torch
import numpy as np
from agent import QNetwork, DQNAgent

class TestQNetwork:
    """Test suite for QNetwork class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        net = QNetwork(input_dim=10, output_dim=5, hidden_dim=32)
        
        # Check layer dimensions
        assert net.input_layer[0].in_features == 10
        assert net.input_layer[0].out_features == 32
        assert net.output_layer.in_features == 32
        assert net.output_layer.out_features == 5
        
    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            QNetwork(input_dim=0, output_dim=5)
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            QNetwork(input_dim=10, output_dim=0)
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            QNetwork(input_dim=10, output_dim=5, hidden_dim=0)
            
    def test_forward_valid(self):
        """Test forward pass with valid inputs."""
        net = QNetwork(input_dim=4, output_dim=3, hidden_dim=8)
        
        # Single sample
        x = torch.randn(4)
        output = net(x)
        assert output.shape == torch.Size([1, 3])  # Output includes batch dimension
        
        # Batch of samples
        x = torch.randn(2, 4)
        output = net(x)
        assert output.shape == torch.Size([2, 3])
        
        # Test training mode (should keep batch dimension)
        net.train()
        x = torch.randn(1, 4)
        output = net(x)
        assert output.shape == torch.Size([1, 3])
        
        # Test eval mode with batch size > 1 (should keep batch dimension)
        net.eval()
        x = torch.randn(2, 4)
        output = net(x)
        assert output.shape == torch.Size([2, 3])
        
        # Test eval mode with single sample (should squeeze output)
        net.eval()
        x = torch.randn(1, 4)
        output = net(x)
        assert output.shape == torch.Size([3])  # No batch dimension
        
    def test_forward_invalid(self):
        """Test forward pass with invalid inputs."""
        net = QNetwork(input_dim=4, output_dim=3)
        
        # Wrong input type
        with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
            net(np.array([1, 2, 3, 4]))
            
        # Wrong input dimension
        with pytest.raises(RuntimeError):
            net(torch.randn(4, 5))  # Wrong input dimension
            
class TestDQNAgent:
    """Test suite for DQNAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create a DQNAgent instance for testing."""
        return DQNAgent(
            state_dim=10,
            action_dim=5,
            hidden_dim=32,
            lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=100,
            device="cpu"
        )
    
    def test_init_valid(self, agent):
        """Test initialization with valid parameters."""
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.1
        assert agent.epsilon_decay == 0.995
        assert agent.target_update_freq == 100
        assert isinstance(agent.online_net, QNetwork)
        assert isinstance(agent.target_net, QNetwork)
        
    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="state_dim and action_dim must be positive"):
            DQNAgent(state_dim=0, action_dim=5)
        with pytest.raises(ValueError, match="state_dim and action_dim must be positive"):
            DQNAgent(state_dim=10, action_dim=0)
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, gamma=1.5)
        with pytest.raises(ValueError, match="epsilon values must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, epsilon_start=1.5)
        with pytest.raises(ValueError, match="epsilon values must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, epsilon_end=1.5)
        with pytest.raises(ValueError, match="epsilon_decay must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, epsilon_decay=0)
        with pytest.raises(ValueError, match="target_update_freq must be positive"):
            DQNAgent(state_dim=10, action_dim=5, target_update_freq=0)
            
    def test_select_action_valid(self, agent):
        """Test action selection with valid inputs."""
        state = np.random.randn(10)
        valid_mask = [True] * 5
        
        # Test greedy action selection (epsilon = 0)
        agent.epsilon = 0
        action = agent.select_action(state, valid_mask)
        assert 0 <= action < 5
        
        # Test random action selection (epsilon = 1)
        agent.epsilon = 1
        action = agent.select_action(state, valid_mask)
        assert 0 <= action < 5
        
        # Test with torch tensor input
        state = torch.randn(10)
        action = agent.select_action(state, valid_mask)
        assert 0 <= action < 5
        
        # Test with partial valid mask
        valid_mask = [True, False, True, False, True]
        action = agent.select_action(state, valid_mask)
        assert valid_mask[action]  # Selected action should be valid
        
    def test_select_action_invalid(self, agent):
        """Test action selection with invalid inputs."""
        state = np.random.randn(10)
        
        # Invalid mask type
        with pytest.raises(TypeError, match="valid_mask must be a numpy array or list"):
            agent.select_action(state, "invalid")
            
        # Invalid mask length
        with pytest.raises(ValueError, match="valid_mask length .* must match action_dim"):
            agent.select_action(state, [True, False])
            
        # Invalid state type
        with pytest.raises(TypeError, match="state must be a numpy array or torch tensor"):
            agent.select_action("invalid", [True] * 5)

        # Invalid state shape
        with pytest.raises(ValueError, match="state dimension .* must match state_dim"):
            agent.select_action(np.random.randn(15), [True] * 5)

        # No valid actions
        with pytest.raises(ValueError, match="At least one action must be valid"):
            agent.select_action(state, [False] * 5)

        # Test with eval mode and invalid state
        with pytest.raises(TypeError, match="state must be a numpy array or torch tensor"):
            agent.select_action("invalid", [True] * 5, eval_mode=True)
        
    def test_learn(self, agent):
        """Test learning from a batch of transitions."""
        batch_size = 4
        batch = {
            'states': torch.randn(batch_size, 10, dtype=torch.float32),  # Use float32 explicitly
            'actions': torch.randint(0, 5, size=(batch_size,)),
            'rewards': torch.randn(batch_size, dtype=torch.float32),
            'next_states': torch.randn(batch_size, 10, dtype=torch.float32),
            'dones': torch.randint(0, 2, size=(batch_size,))
        }
        
        # Initial statistics
        initial_steps = agent.train_steps
        initial_updates = agent.updates
        
        # Set train_steps to trigger target network update
        agent.train_steps = agent.target_update_freq - 1
        
        # Perform learning step
        loss = agent.learn(batch)
        
        # Check that statistics were updated
        assert agent.train_steps == agent.target_update_freq  # One step taken
        assert isinstance(loss, float)
        assert loss >= 0
        
        # Check target network update
        assert agent.updates == initial_updates + 1  # Should have updated

    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent state."""
        # Generate a save path
        save_path = tmp_path / "agent.pt"
        
        # Save initial state
        initial_state_dict = agent.online_net.state_dict()
        initial_stats = agent.get_statistics()
        agent.save(str(save_path))
        
        # Modify agent state
        agent.train_steps += 100
        agent.epsilon = 0.5
        
        # Load saved state
        agent.load(str(save_path))
        
        # Check that state was restored
        loaded_state_dict = agent.online_net.state_dict()
        loaded_stats = agent.get_statistics()
        
        assert torch.all(torch.eq(initial_state_dict['input_layer.0.weight'],
                                loaded_state_dict['input_layer.0.weight']))
        assert initial_stats['train_steps'] == loaded_stats['train_steps']
        assert initial_stats['epsilon'] == loaded_stats['epsilon']
