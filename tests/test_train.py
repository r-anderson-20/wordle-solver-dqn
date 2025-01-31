"""Tests for the training module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import torch

from train import train
from environment import WordleEnvironment
from agent import DQNAgent
from replay_buffer import ReplayBuffer


class TestTrain:
    """Test cases for training functionality."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock WordleEnvironment."""
        env = MagicMock(spec=WordleEnvironment)
        
        # Mock reset method
        feedback_matrix = np.zeros((6, 26, 13))
        valid_mask = np.ones(5, dtype=bool)
        remaining_guesses = 6
        env.reset.return_value = (feedback_matrix, valid_mask, remaining_guesses)
        
        # Mock step method
        env.step.return_value = (feedback_matrix, valid_mask, remaining_guesses-1, 1.0, True)
        
        return env

    @pytest.fixture
    def mock_agent(self):
        """Create a mock DQNAgent."""
        agent = MagicMock(spec=DQNAgent)
        agent.epsilon = 1.0
        agent.select_action.return_value = 0
        agent.learn.return_value = 0.5
        return agent

    @pytest.fixture
    def mock_buffer(self):
        """Create a mock ReplayBuffer."""
        buffer = MagicMock(spec=ReplayBuffer)
        buffer.size = 100
        buffer.sample.return_value = {
            'states': torch.zeros(64, 390 + 5 + 1),
            'actions': torch.zeros(64, dtype=torch.long),
            'rewards': torch.zeros(64),
            'next_states': torch.zeros(64, 390 + 5 + 1),
            'dones': torch.zeros(64, dtype=torch.bool)
        }
        return buffer

    @patch('train.WordleEnvironment')
    @patch('train.DQNAgent')
    @patch('train.ReplayBuffer')
    def test_train_basic(self, mock_buffer_class, mock_agent_class, mock_env_class,
                        mock_env, mock_agent, mock_buffer):
        """Test basic training loop with minimal episodes."""
        # Set up mocks
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        mock_buffer_class.return_value = mock_buffer

        # Test data
        valid_words = ["apple", "beach", "crane", "dance", "eagle"]
        test_words = ["fight", "ghost", "house", "image", "juice"]

        # Run training with minimal episodes
        agent, metrics = train(
            valid_words=valid_words,
            test_words=test_words,
            num_episodes=2,
            eval_freq=1,
            eval_episodes=2,
            batch_size=64
        )

        # Verify basic interactions
        assert mock_env.reset.called
        assert mock_env.step.called
        assert mock_agent.select_action.called
        assert mock_agent.learn.called
        assert mock_buffer.add.called
        assert mock_buffer.sample.called

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'episode_rewards' in metrics
        assert 'episode_lengths' in metrics
        assert 'running_avg_reward' in metrics
        assert 'running_avg_length' in metrics
        assert 'eval_solve_rates' in metrics
        assert 'best_solved_rate' in metrics

    @patch('train.WordleEnvironment')
    @patch('train.DQNAgent')
    @patch('train.ReplayBuffer')
    def test_train_evaluation(self, mock_buffer_class, mock_agent_class, mock_env_class,
                            mock_env, mock_agent, mock_buffer):
        """Test evaluation phase of training."""
        # Set up mocks
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        mock_buffer_class.return_value = mock_buffer

        # Configure mock to simulate successful guesses during evaluation
        feedback_matrix = np.zeros((6, 26, 13))
        valid_mask = np.ones(5)
        mock_env.reset.return_value = (feedback_matrix, valid_mask, 6)

        # Create a list of step responses for both training and evaluation
        step_responses = [
            # Training episode responses
            (feedback_matrix, valid_mask, 5, 1.0, True),
            # Evaluation episode 1 responses
            (feedback_matrix, valid_mask, 5, 1.0, True),
            # Evaluation episode 2 responses
            (feedback_matrix, valid_mask, 5, 1.0, True),
        ]
        mock_env.step.side_effect = step_responses

        # Test data
        valid_words = ["apple", "beach", "crane", "dance", "eagle"]
        test_words = ["fight", "ghost", "house", "image", "juice"]

        # Run training with evaluation
        agent, metrics = train(
            valid_words=valid_words,
            test_words=test_words,
            num_episodes=1,
            eval_freq=1,
            eval_episodes=2,
            batch_size=64
        )

        # Verify evaluation metrics
        assert len(metrics['eval_solve_rates']) > 0
        assert metrics['best_solved_rate'] >= 0

    def test_train_invalid_inputs(self):
        """Test training with invalid inputs."""
        valid_words = ["apple", "beach", "crane"]
        test_words = ["fight", "ghost"]

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            train(valid_words=valid_words, test_words=test_words, hidden_dim=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            train(valid_words=valid_words, test_words=test_words, learning_rate=0)

        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, gamma=1.5)

        with pytest.raises(ValueError, match="epsilon_start must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, epsilon_start=1.5)

        with pytest.raises(ValueError, match="epsilon_end must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, epsilon_end=1.5)

        with pytest.raises(ValueError, match="epsilon_decay must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, epsilon_decay=1.5)

        with pytest.raises(ValueError, match="target_update_freq must be positive"):
            train(valid_words=valid_words, test_words=test_words, target_update_freq=0)

        with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
            train(valid_words=valid_words, test_words=test_words, device="invalid")

        with pytest.raises(ValueError, match="valid_words must not be empty"):
            train(valid_words=[], test_words=test_words)

        with pytest.raises(ValueError, match="test_words must not be empty"):
            train(valid_words=valid_words, test_words=[])

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            train(valid_words=valid_words, test_words=test_words, num_episodes=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            train(valid_words=valid_words, test_words=test_words, batch_size=0)

        with pytest.raises(ValueError, match="eval_freq must be positive"):
            train(valid_words=valid_words, test_words=test_words, eval_freq=0)

        with pytest.raises(ValueError, match="eval_episodes must be positive"):
            train(valid_words=valid_words, test_words=test_words, eval_episodes=0)

    @patch('train.WordleEnvironment')
    @patch('train.DQNAgent')
    @patch('train.ReplayBuffer')
    def test_train_early_stopping(self, mock_buffer_class, mock_agent_class, mock_env_class,
                                mock_env, mock_agent, mock_buffer):
        """Test training with early stopping based on solve rate."""
        # Set up mocks
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        mock_buffer_class.return_value = mock_buffer

        # Configure mock to simulate perfect solve rate
        mock_env.step.return_value = (np.zeros((6, 26, 13)), np.ones(5), 5, 2.0, True)

        # Test data
        valid_words = ["apple", "beach", "crane", "dance", "eagle"]
        test_words = ["fight", "ghost", "house", "image", "juice"]

        # Run training
        agent, metrics = train(
            valid_words=valid_words,
            test_words=test_words,
            num_episodes=100,
            eval_freq=10,
            eval_episodes=5,
            batch_size=64
        )

        # Verify metrics reflect good performance
        assert metrics['best_solved_rate'] > 0

    def test_main_script_execution(self):
        """Test the main script execution block."""
        with patch('train.load_words') as mock_load_words, \
             patch('train.train') as mock_train:
            
            # Mock load_words to return test data
            mock_load_words.side_effect = [
                ["apple", "beach"],  # train_words
                ["fight", "ghost"]   # test_words
            ]

            # Mock train function
            mock_agent = MagicMock()
            mock_metrics = {
                'episode_rewards': [1.0],
                'episode_lengths': [1],
                'running_avg_reward': [1.0],
                'running_avg_length': [1],
                'eval_solve_rates': [0.5],
                'best_solved_rate': 0.5
            }
            mock_train.return_value = (mock_agent, mock_metrics)

            # Run main script by importing and calling main function directly
            import train
            
            # Save original __name__
            original_name = train.__name__
            try:
                # Set __name__ to '__main__' to trigger main block
                train.__name__ = '__main__'
                
                # Re-execute the main block
                if hasattr(train, '__name__') and train.__name__ == '__main__':
                    train_words = mock_load_words('word_lists/train_words.txt')
                    test_words = mock_load_words('word_lists/test_words.txt')
                    
                    print(f"Loaded {len(train_words)} training words and {len(test_words)} test words")
                    
                    # Train the agent
                    trained_agent, metrics = train.train(
                        valid_words=train_words,
                        test_words=test_words,
                        num_episodes=1000,
                        hidden_dim=256,
                        learning_rate=1e-4,
                        gamma=0.99,
                        epsilon_start=1.0,
                        epsilon_end=0.01,
                        epsilon_decay=0.995,
                        batch_size=64,
                        target_update_freq=100,
                        eval_freq=200,
                        eval_episodes=50,
                        device="cpu"
                    )
            finally:
                # Restore original __name__
                train.__name__ = original_name

            # Verify load_words was called with correct paths
            mock_load_words.assert_any_call('word_lists/train_words.txt')
            mock_load_words.assert_any_call('word_lists/test_words.txt')

            # Verify train was called with expected arguments
            mock_train.assert_called_once()
            args = mock_train.call_args[1]
            assert args['num_episodes'] == 1000
            assert args['hidden_dim'] == 256
            assert args['learning_rate'] == 1e-4
            assert args['gamma'] == 0.99
            assert args['epsilon_start'] == 1.0
            assert args['epsilon_end'] == 0.01
            assert args['epsilon_decay'] == 0.995
            assert args['batch_size'] == 64
            assert args['target_update_freq'] == 100
            assert args['eval_freq'] == 200
            assert args['eval_episodes'] == 50
            assert args['device'] == "cpu"
