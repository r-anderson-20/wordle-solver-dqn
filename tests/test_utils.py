"""
Tests for utility functions in utils.py.
"""

import pytest
import numpy as np
import os
import tempfile
import io
import sys
from utils import load_words, flatten_state, colorize_feedback, print_known_letters

class TestLoadWords:
    """Tests for the load_words function."""
    
    def test_load_words_success(self, temp_word_file, small_word_list):
        """Test loading words from a valid file."""
        words = load_words(temp_word_file)
        assert words == small_word_list
        assert all(len(word) == 5 for word in words)
    
    def test_load_words_empty_file(self):
        """Test loading words from an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            pass
        try:
            words = load_words(f.name)
            assert words == []
        finally:
            os.unlink(f.name)
    
    def test_load_words_file_not_found(self):
        """Test loading words from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_words("nonexistent_file.txt")
            
    def test_load_words_with_invalid_length(self):
        """Test loading words with invalid length."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("word\n")  # 4 letters
            f.write("toolong\n")  # 7 letters
            f.write("valid\n")  # 5 letters
        try:
            words = load_words(f.name)
            assert words == ["valid"]
        finally:
            os.unlink(f.name)
            
    def test_load_words_with_whitespace(self):
        """Test loading words with whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("  apple  \n")  # Leading/trailing spaces
            f.write("beach\n")
            f.write("\n")  # Empty line
            f.write("crane  ")  # No newline at end
        try:
            words = load_words(f.name)
            assert words == ["apple", "beach", "crane"]
        finally:
            os.unlink(f.name)

class TestFlattenState:
    """Tests for the flatten_state function."""
    
    def test_flatten_state_basic(self, feedback_matrix, valid_mask):
        """Test basic state flattening."""
        remaining_guesses = 6
        flattened = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
        
        # Check dimensions
        expected_length = 5 * 26 * 3 + len(valid_mask) + 1
        assert len(flattened) == expected_length
        
        # Check remaining_guesses is last element
        assert flattened[-1] == remaining_guesses
        
        # Check feedback matrix is properly flattened
        feedback_flat = flattened[:5*26*3]
        assert np.array_equal(feedback_flat, feedback_matrix.flatten())
        
        # Check valid mask is included
        mask_start = 5*26*3
        mask_end = mask_start + len(valid_mask)
        assert np.array_equal(flattened[mask_start:mask_end], valid_mask.astype(np.float32))
    
    def test_flatten_state_types(self, feedback_matrix, valid_mask):
        """Test output types of flatten_state."""
        flattened = flatten_state(feedback_matrix, valid_mask, 6)
        assert flattened.dtype == np.float32
        
    def test_flatten_state_zero_guesses(self, feedback_matrix, valid_mask):
        """Test with zero remaining guesses."""
        flattened = flatten_state(feedback_matrix, valid_mask, 0)
        assert flattened[-1] == 0
        
    def test_flatten_state_empty_mask(self, feedback_matrix):
        """Test with empty valid mask."""
        empty_mask = np.array([], dtype=bool)
        flattened = flatten_state(feedback_matrix, empty_mask, 6)
        assert len(flattened) == 5 * 26 * 3 + 1  # No mask elements

class TestColorizeFeedback:
    """Tests for the colorize_feedback function."""
    
    @pytest.mark.parametrize("guess,feedback_string,expected_colors", [
        ("apple", "GBBBB", ["\033[92m", "\033[90m", "\033[90m", "\033[90m", "\033[90m"]),  # Green, Gray x4
        ("beach", "YYYYY", ["\033[93m"] * 5),  # All Yellow
        ("crane", "GGGGG", ["\033[92m"] * 5),  # All Green
        ("dance", "BBBBB", ["\033[90m"] * 5),  # All Gray
    ])
    def test_colorize_feedback_patterns(self, guess, feedback_string, expected_colors):
        """Test different feedback patterns."""
        result = colorize_feedback(guess, feedback_string)
        
        # Check each character has correct color code
        for i, color in enumerate(expected_colors):
            assert color in result.split()[i]
            assert guess[i] in result.split()[i]
            assert "\033[0m" in result.split()[i]  # Reset code
            
    def test_colorize_feedback_invalid_feedback(self):
        """Test with invalid feedback string."""
        with pytest.raises(ValueError):
            colorize_feedback("apple", "INVALID")
            
    def test_colorize_feedback_length_mismatch(self):
        """Test with mismatched lengths."""
        with pytest.raises(ValueError):
            colorize_feedback("apple", "GX")  # Too short
        with pytest.raises(ValueError):
            colorize_feedback("apple", "GXXXXX")  # Too long

class TestPrintKnownLetters:
    """Tests for the print_known_letters function."""
    
    def test_print_known_letters(self, feedback_matrix):
        """Test printing known letters information."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        print_known_letters(feedback_matrix)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Check output contains expected information
        assert "Known information:" in output
        assert "Correct positions:" in output
        assert "a at position 1" in output.lower()  # From feedback_matrix fixture
        assert "Correct letters:" in output
        assert "b" in output.lower()  # From feedback_matrix fixture
        assert "Incorrect letters:" in output
        assert "c" in output.lower()  # From feedback_matrix fixture
        
    def test_print_known_letters_empty(self):
        """Test printing known letters with empty feedback matrix."""
        empty_matrix = np.zeros((5, 26, 3))
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        print_known_letters(empty_matrix)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Check output shows no known information
        assert "Known information:" in output
        assert "Correct positions:" not in output  # No correct positions
        assert "Correct letters:" not in output    # No correct letters
        assert "Incorrect letters:" not in output  # No incorrect letters
        
    def test_print_known_letters_all_types(self):
        """Test printing with all types of feedback."""
        matrix = np.zeros((5, 26, 3))
        # Add one of each type
        matrix[0, 0, 0] = 1  # 'a' correct at pos 0
        matrix[1, 1, 1] = 1  # 'b' present at pos 1
        matrix[2, 2, 2] = 1  # 'c' absent at pos 2
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        print_known_letters(matrix)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "a at position 1" in output.lower()
        assert "b" in output.lower()
        assert "c" in output.lower()
