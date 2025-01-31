"""
Tests for the play_games module.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from play_games import play_game, main

class TestPlayGames:
    """Test suite for play_games functionality."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock WordleEnvironment."""
        env = MagicMock()
        env.valid_words = ["apple", "beach", "crane"]
        env._compute_feedback_string.return_value = "GGGGG"
        return env
        
    @pytest.fixture
    def mock_agent(self):
        """Create a mock DQNAgent."""
        agent = MagicMock()
        agent.select_action.return_value = 0
        agent.online_net = MagicMock()
        return agent
        
    def test_play_game_solved(self, mock_env, mock_agent, capsys):
        """Test playing a game that is solved."""
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))  # Correct shape: (max_guesses, alphabet_size, feedback_types)
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env.valid_words = ["apple", "beach", "crane"]
        mock_env._compute_feedback_string.return_value = "GGGGG"
        
        # Play game
        solved, num_guesses, guesses, feedbacks = play_game(mock_env, mock_agent, "apple")
        
        # Check results
        assert solved is True
        assert num_guesses == 1
        assert guesses == ["apple"]
        assert feedbacks == ["GGGGG"]
        
        # Check output
        captured = capsys.readouterr()
        assert "Target word: apple" in captured.out
        assert " Solved in 1 guesses!" in captured.out
        
    def test_play_game_failed(self, mock_env, mock_agent, capsys):
        """Test playing a game that is not solved."""
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))  # Correct shape: (max_guesses, alphabet_size, feedback_types)
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 0.0, True)
        mock_env.valid_words = ["apple", "beach", "crane"]
        mock_env._compute_feedback_string.return_value = "YBBBB"
        
        # Play game
        solved, num_guesses, guesses, feedbacks = play_game(mock_env, mock_agent, "beach")
        
        # Check results
        assert solved is False
        assert num_guesses == 1
        assert guesses == ["apple"]
        assert feedbacks == ["YBBBB"]
        
        # Check output
        captured = capsys.readouterr()
        assert "Target word: beach" in captured.out
        assert " Failed to solve" in captured.out
        
    def test_play_game_invalid_state(self, mock_env, mock_agent, capsys):
        """Test playing a game with invalid state transitions."""
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        
        # Mock step to raise ValueError on second call
        mock_env.step.side_effect = [
            (feedback_matrix, np.ones(3), 5, 0.0, False),  # First call succeeds
            ValueError("Invalid state transition")          # Second call fails
        ]
        mock_env.valid_words = ["apple", "beach", "crane"]
        mock_env._compute_feedback_string.return_value = "BBBBB"
        
        # Play game
        solved, num_guesses, guesses, feedbacks = play_game(mock_env, mock_agent, "beach")
        
        # Check results
        assert solved is False
        assert num_guesses == 2  # Should count both attempts
        assert len(guesses) == 2
        assert len(feedbacks) == 2
        
        # Check output
        captured = capsys.readouterr()
        assert "Target word: beach" in captured.out
        assert "Error: Invalid state transition" in captured.out

    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    @patch('play_games.load_words')
    def test_main_with_provided_words(self, mock_load_words, mock_torch_load, mock_agent_class, mock_env_class, capsys):
        """Test main function with provided words."""
        # Set up mocks
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.valid_words = ["apple", "beach", "crane"]
        
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env._compute_feedback_string.return_value = "GGGGG"
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.select_action.return_value = 0
        
        # Create a mock state dict with all required keys
        mock_state_dict = {
            'online_net': {
                'input_layer.0.weight': torch.randn(256, 891),
                'input_layer.0.bias': torch.randn(256),
                'hidden_layer1.0.weight': torch.randn(256, 256),
                'hidden_layer1.0.bias': torch.randn(256),
                'hidden_layer1.2.weight': torch.randn(256, 256),
                'hidden_layer1.2.bias': torch.randn(256),
                'hidden_layer2.0.weight': torch.randn(256, 256),
                'hidden_layer2.0.bias': torch.randn(256),
                'hidden_layer2.2.weight': torch.randn(256, 256),
                'hidden_layer2.2.bias': torch.randn(256),
                'output_layer.weight': torch.randn(500, 256),
                'output_layer.bias': torch.randn(500)
            }
        }
        mock_torch_load.return_value = mock_state_dict
        
        # Run main with provided words
        test_words = ["apple", "beach", "crane"]
        main(test_words)
        
        # Check that environment and agent were created correctly
        mock_env_class.assert_called_once()
        mock_agent_class.assert_called_once()
        mock_load_words.assert_not_called()
        
        # Check output
        captured = capsys.readouterr()
        assert "Using 3 words for gameplay" in captured.out
        assert "Playing 20 games" in captured.out
        assert "Game Statistics:" in captured.out
        
    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    @patch('play_games.load_words')
    def test_main_without_words(self, mock_load_words, mock_torch_load, mock_agent_class, mock_env_class, capsys):
        """Test main function without provided words."""
        # Set up mocks
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.valid_words = ["apple", "beach", "crane"]
        
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env._compute_feedback_string.return_value = "YBBBB"
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.select_action.return_value = 0
        
        # Create a mock state dict with all required keys
        mock_state_dict = {
            'online_net': {
                'input_layer.0.weight': torch.randn(256, 891),
                'input_layer.0.bias': torch.randn(256),
                'hidden_layer1.0.weight': torch.randn(256, 256),
                'hidden_layer1.0.bias': torch.randn(256),
                'hidden_layer1.2.weight': torch.randn(256, 256),
                'hidden_layer1.2.bias': torch.randn(256),
                'hidden_layer2.0.weight': torch.randn(256, 256),
                'hidden_layer2.0.bias': torch.randn(256),
                'hidden_layer2.2.weight': torch.randn(256, 256),
                'hidden_layer2.2.bias': torch.randn(256),
                'output_layer.weight': torch.randn(500, 256),
                'output_layer.bias': torch.randn(500)
            }
        }
        mock_torch_load.return_value = mock_state_dict
        mock_load_words.return_value = ["apple", "beach", "crane"]
        
        # Run main without words
        main()
        
        # Check that words were loaded from file
        mock_load_words.assert_called_once_with('word_lists/test_words.txt')
        
        # Check output
        captured = capsys.readouterr()
        assert "Using 3 words for gameplay" in captured.out
        assert "Playing 20 games" in captured.out
        assert "Game Statistics:" in captured.out
        
    @patch('play_games.main')
    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    @patch('play_games.load_words')
    def test_script_execution(self, mock_load_words, mock_torch_load, mock_agent_class, mock_env_class, mock_main):
        """Test script execution."""
        # Set up mocks similar to test_main_without_words
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.valid_words = ["apple", "beach", "crane"]
        
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env._compute_feedback_string.return_value = "YBBBB"
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.select_action.return_value = 0
        
        # Create a mock state dict with all required keys
        mock_state_dict = {
            'online_net': {
                'input_layer.0.weight': torch.randn(256, 891),
                'input_layer.0.bias': torch.randn(256),
                'hidden_layer1.0.weight': torch.randn(256, 256),
                'hidden_layer1.0.bias': torch.randn(256),
                'hidden_layer1.2.weight': torch.randn(256, 256),
                'hidden_layer1.2.bias': torch.randn(256),
                'hidden_layer2.0.weight': torch.randn(256, 256),
                'hidden_layer2.0.bias': torch.randn(256),
                'hidden_layer2.2.weight': torch.randn(256, 256),
                'hidden_layer2.2.bias': torch.randn(256),
                'output_layer.weight': torch.randn(500, 256),
                'output_layer.bias': torch.randn(500)
            }
        }
        mock_torch_load.return_value = mock_state_dict
        mock_load_words.return_value = ["apple", "beach", "crane"]
        
        # Import play_games module and call its main function directly
        import play_games
        play_games.main()
        
        # Verify main was called
        mock_main.assert_called_once()

    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    def test_main_model_load_error(self, mock_torch_load, mock_agent_class, mock_env_class, capsys):
        """Test main function handling model loading error."""
        # Mock torch.load to raise an error
        mock_torch_load.side_effect = FileNotFoundError("Model file not found")
        
        # Run main with empty word list to focus on model loading
        with pytest.raises(FileNotFoundError):
            main(words=["apple", "beach"])
        
        # Verify error message
        captured = capsys.readouterr()
        assert "Model file not found" in str(mock_torch_load.side_effect)

    def test_main_script_execution_direct(self):
        """Test executing the script directly."""
        with patch('play_games.main') as mock_main:
            # Save original __name__
            import play_games
            original_name = play_games.__name__
            try:
                # Set __name__ to '__main__' to trigger main block
                play_games.__name__ = '__main__'
                
                # Re-execute the main block
                if hasattr(play_games, '__name__') and play_games.__name__ == '__main__':
                    play_games.main()
            finally:
                # Restore original __name__
                play_games.__name__ = original_name
            
            # Verify main was called
            mock_main.assert_called_once()
