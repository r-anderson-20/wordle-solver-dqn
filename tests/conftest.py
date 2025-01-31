"""
Shared test fixtures for the Wordle DQN solver test suite.
"""

import pytest
import numpy as np
import os
import tempfile

@pytest.fixture
def small_word_list():
    """A small list of 5-letter words for testing."""
    return ["apple", "beach", "crane", "dance", "eagle"]

@pytest.fixture
def temp_word_file(small_word_list):
    """Creates a temporary file with test words."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for word in small_word_list:
            f.write(word + '\n')
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def feedback_matrix():
    """A sample feedback matrix for testing."""
    matrix = np.zeros((5, 26, 3))  # 5 positions, 26 letters, 3 feedback types
    # Add some sample feedback
    matrix[0, 0, 0] = 1  # 'a' is correct at position 0
    matrix[1, 1, 1] = 1  # 'b' is present but wrong position at position 1
    matrix[2, 2, 2] = 1  # 'c' is absent at position 2
    return matrix

@pytest.fixture
def valid_mask(small_word_list):
    """A sample valid mask for testing."""
    return np.ones(len(small_word_list), dtype=bool)
