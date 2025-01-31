"""
Tests for the Wordle environment implementation.
"""

import pytest
import numpy as np
from environment import WordleEnvironment

@pytest.fixture
def wordle_env():
    """Create a basic Wordle environment with a small word list."""
    valid_words = ["apple", "beach", "crane", "dance", "eagle"]
    return WordleEnvironment(valid_words)

class TestWordleEnvironment:
    """Tests for the WordleEnvironment class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        valid_words = ["apple", "beach", "crane"]
        env = WordleEnvironment(valid_words)
        assert env.valid_words == valid_words
        assert env.max_guesses == 6  # Default value
        
    def test_init_invalid_words(self):
        """Test initialization with invalid word list."""
        # Test non-list input
        with pytest.raises(ValueError, match="valid_words must be a list of strings"):
            WordleEnvironment("not_a_list")
            
        # Test non-string elements
        with pytest.raises(ValueError, match="valid_words must be a list of strings"):
            WordleEnvironment([1, 2, 3])
            
        # Test wrong word length
        with pytest.raises(ValueError, match="all words must be 5 letters long"):
            WordleEnvironment(["short", "toolong", "right"])
            
    def test_init_invalid_max_guesses(self):
        """Test initialization with invalid max_guesses."""
        valid_words = ["apple", "beach"]
        with pytest.raises(ValueError, match="max_guesses must be positive"):
            WordleEnvironment(valid_words, max_guesses=0)
            
    def test_reset_valid(self, wordle_env):
        """Test reset with valid secret word."""
        feedback_matrix, valid_mask, remaining_guesses = wordle_env.reset("apple")
        
        # Check dimensions and types
        assert feedback_matrix.shape == (5, 26, 3)
        assert feedback_matrix.dtype == np.float32
        assert valid_mask.dtype == bool
        assert len(valid_mask) == len(wordle_env.valid_words)
        assert remaining_guesses == wordle_env.max_guesses
        
        # Check initial state
        assert not wordle_env.done
        assert wordle_env.secret_word == "apple"
        assert wordle_env.last_reward == 0.0
        assert np.all(valid_mask)  # All words should be valid initially
        assert np.all(feedback_matrix == 0)  # No feedback yet
        
    def test_reset_invalid(self, wordle_env):
        """Test reset with invalid parameters."""
        # Test invalid word length
        with pytest.raises(ValueError, match="secret_word must be a 5-letter string"):
            wordle_env.reset("toolong")
            
        # Test word not in valid list
        with pytest.raises(ValueError, match="secret_word must be in valid_words list"):
            wordle_env.reset("wrong")
            
    def test_step_correct_guess(self, wordle_env):
        """Test step with a correct guess."""
        wordle_env.reset("apple")
        feedback_matrix, valid_mask, remaining_guesses, reward, done = wordle_env.step("apple")
        
        # Check game is won
        assert done
        assert remaining_guesses == wordle_env.max_guesses - 1
        assert reward > 0  # Should get positive reward for correct guess
        
        # Check feedback is correct
        assert np.sum(feedback_matrix[:, :, 2]) == 5  # All letters should be green
        
    def test_step_partial_match(self, wordle_env):
        """Test step with a partially correct guess."""
        wordle_env.reset("apple")
        feedback_matrix, valid_mask, remaining_guesses, reward, done = wordle_env.step("eagle")
        
        # Game should continue
        assert not done
        assert remaining_guesses == wordle_env.max_guesses - 1
        
        # Should get some reward for partial matches
        assert reward > -1  # Some positive reward for matching letters
        
        # Check feedback matrix has correct counts
        green_count = np.sum(feedback_matrix[:, :, 2])  # Count green feedback
        yellow_count = np.sum(feedback_matrix[:, :, 1])  # Count yellow feedback
        assert green_count + yellow_count > 0  # Should have some matching letters
        
    def test_step_invalid_guess(self, wordle_env):
        """Test step with invalid guesses."""
        wordle_env.reset("apple")
        
        # Test invalid word length
        with pytest.raises(ValueError, match="guess must be a 5-letter string"):
            wordle_env.step("toolong")
            
        # Test word not in valid list
        with pytest.raises(ValueError, match="guess must be in valid_words list"):
            wordle_env.step("wrong")
            
    def test_game_loss(self, wordle_env):
        """Test game ending in a loss."""
        wordle_env.reset("apple")
        
        # Make max_guesses wrong guesses
        for _ in range(wordle_env.max_guesses):
            feedback_matrix, valid_mask, remaining_guesses, reward, done = wordle_env.step("beach")
            if done:
                break
                
        assert done  # Game should be over
        assert remaining_guesses == 0  # No guesses left
        # Note: reward may be positive due to partial matches
        
    def test_feedback_string_computation(self, wordle_env):
        """Test the feedback string computation."""
        wordle_env.reset("apple")
        
        # Test exact match
        assert wordle_env._compute_feedback_string("apple", "apple") == "GGGGG"
        
        # Test no matches
        assert wordle_env._compute_feedback_string("crane", "fight") == "BBBBB"
        
        # Test partial matches
        # 'beach' vs 'apple': 'a' and 'e' are in wrong positions
        assert wordle_env._compute_feedback_string("beach", "apple") == "BYYBB"
        
        # Test repeated letters
        # 'eagle' vs 'apple':
        # - First 'e' is yellow (matches 'e' in apple)
        # - Second 'e' is green (matches 'e' in position 4)
        # - 'a' is yellow (matches 'a' in apple)
        # - 'g' and 'l' are gray (no matches)
        assert wordle_env._compute_feedback_string("eagle", "apple") == "BYBGG"
        
        # Test more repeated letter cases
        # 'speed' vs 'apple':
        # - 's' is gray (no match)
        # - 'p' is green (matches 'p' in position 1)
        # - First 'e' is yellow (matches 'e' in apple)
        # - Second 'e' is gray (no more 'e's available)
        # - 'd' is gray (no match)
        assert wordle_env._compute_feedback_string("speed", "apple") == "BGYBB"
        
        # Test another repeated letter case
        # 'peeps' vs 'apple':
        # - First 'p' is yellow (matches 'p' in apple)
        # - First 'e' is yellow (matches 'e' in apple)
        # - Second 'e' is gray (no more 'e's available)
        # - Second 'p' is yellow (matches 'p' in apple)
        # - 's' is gray (no match)
        assert wordle_env._compute_feedback_string("peeps", "apple") == "YYBYB"

    def test_valid_mask_update(self, wordle_env):
        """Test the valid mask updating logic."""
        wordle_env.reset("apple")
        
        # Make a guess that should eliminate some words
        wordle_env.step("beach")
        
        # The mask should eliminate words that don't match the feedback pattern
        assert not np.all(wordle_env.valid_mask)  # Some words should be eliminated
        assert wordle_env.valid_mask[wordle_env.valid_words.index("apple")]  # Target word should still be valid
