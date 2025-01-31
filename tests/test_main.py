"""
Tests for the main script functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import argparse
import torch
from main import parse_args, main

class TestMain:
    """Test suite for main script functionality."""
    
    def test_parse_args_train(self, monkeypatch):
        """Test argument parsing for train mode."""
        # Mock command line arguments
        test_args = ["--mode", "train",
                    "--model_path", "test_model.pth",
                    "--train_words", "train.txt",
                    "--test_words", "test.txt",
                    "--num_episodes", "100",
                    "--hidden_dim", "128",
                    "--batch_size", "32",
                    "--learning_rate", "0.001",
                    "--device", "cpu"]
        monkeypatch.setattr("sys.argv", ["main.py"] + test_args)
        
        args = parse_args()
        assert args.mode == "train"
        assert args.model_path == "test_model.pth"
        assert args.train_words == "train.txt"
        assert args.test_words == "test.txt"
        assert args.num_episodes == 100
        assert args.hidden_dim == 128
        assert args.batch_size == 32
        assert args.learning_rate == 0.001
        assert args.device == "cpu"
        
    def test_parse_args_test(self, monkeypatch):
        """Test argument parsing for test mode."""
        test_args = ["--mode", "test",
                    "--test_words", "test.txt"]
        monkeypatch.setattr("sys.argv", ["main.py"] + test_args)
        
        args = parse_args()
        assert args.mode == "test"
        assert args.test_words == "test.txt"
        
    def test_parse_args_invalid_mode(self, monkeypatch):
        """Test argument parsing with invalid mode."""
        test_args = ["--mode", "invalid"]
        monkeypatch.setattr("sys.argv", ["main.py"] + test_args)
        
        with pytest.raises(SystemExit):
            parse_args()
            
    @patch('main.load_words')
    @patch('main.train')
    @patch('torch.save')
    def test_main_train_mode(self, mock_save, mock_train, mock_load_words, monkeypatch):
        """Test main function in train mode."""
        # Mock command line arguments
        test_args = ["--mode", "train",
                    "--model_path", "test_model.pth",
                    "--device", "cpu"]
        monkeypatch.setattr("sys.argv", ["main.py"] + test_args)
        
        # Mock dependencies
        mock_load_words.return_value = ["word1", "word2"]
        mock_agent = MagicMock()
        mock_agent.online_net = MagicMock()
        mock_train.return_value = (mock_agent, None)
        
        # Run main
        main()
        
        # Verify function calls
        assert mock_load_words.call_count == 2  # Called for both train and test words
        mock_train.assert_called_once()
        mock_save.assert_called_once_with(
            mock_agent.online_net.state_dict(),
            "test_model.pth"
        )
        
    @patch('main.load_words')
    @patch('main.play_games')
    def test_main_test_mode(self, mock_play_games, mock_load_words, monkeypatch):
        """Test main function in test mode."""
        # Mock command line arguments
        test_args = ["--mode", "test",
                    "--test_words", "test.txt"]
        monkeypatch.setattr("sys.argv", ["main.py"] + test_args)
        
        # Mock dependencies
        mock_load_words.return_value = ["word1", "word2"]
        
        # Run main
        main()
        
        # Verify function calls
        mock_load_words.assert_called_once_with("test.txt")
        mock_play_games.assert_called_once_with(["word1", "word2"])

    def test_script_execution(self, monkeypatch):
        """Test script execution by verifying argument parsing."""
        test_args = ["main.py", "--mode", "test", "--test_words", "word_lists/test_words.txt"]
        monkeypatch.setattr("sys.argv", test_args)
        
        # Import main and call parse_args directly
        from main import parse_args
        args = parse_args()
        
        # Verify arguments were parsed correctly
        assert args.mode == "test"
        assert args.test_words == "word_lists/test_words.txt"
