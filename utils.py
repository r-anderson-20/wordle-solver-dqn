"""
Utility functions for the Wordle DQN solver.
Contains common functions used across different modules.
"""

import numpy as np

def flatten_state(feedback_matrix, valid_mask, remaining_guesses):
    """
    Convert the environment state into a 1D vector for neural network input.
    
    Args:
        feedback_matrix (np.ndarray): 5x26x3 matrix of letter feedback
            - First dimension (5): Letter positions
            - Second dimension (26): Letters of the alphabet
            - Third dimension (3): Feedback type [correct, present, absent]
        valid_mask (np.ndarray): Boolean mask of valid words in dictionary
        remaining_guesses (int): Number of guesses remaining
        
    Returns:
        np.ndarray: Flattened state vector concatenating:
            - Flattened feedback matrix (5*26*3 = 390 elements)
            - Valid word mask (dictionary_size elements)
            - Remaining guesses (1 element)
    """
    # Ensure inputs are numpy arrays
    feedback_matrix = np.asarray(feedback_matrix)
    valid_mask = np.asarray(valid_mask)
    
    # Flatten and convert to float32
    feedback_flat = feedback_matrix.reshape(-1).astype(np.float32)
    valid_mask_float = valid_mask.astype(np.float32)
    remaining_guesses_float = np.array([float(remaining_guesses)], dtype=np.float32)
    
    return np.concatenate([feedback_flat, valid_mask_float, remaining_guesses_float])

def load_words(filename):
    """
    Load words from a text file, one word per line.
    Only includes 5-letter words, stripping whitespace.
    
    Args:
        filename (str): Path to the word list file
        
    Returns:
        list[str]: List of 5-letter words from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f if line.strip() and len(line.strip()) == 5]

def colorize_feedback(guess, feedback_string):
    """
    Convert feedback string to colored output.

    Args:
        guess (str): The guessed word
        feedback_string (str): The feedback string from the environment
                             Must be 5 characters long, using G/Y/B for Green/Yellow/Black

    Returns:
        str: The colored feedback string

    Raises:
        ValueError: If feedback_string is invalid or length doesn't match guess
    """
    if len(guess) != len(feedback_string):
        raise ValueError(f"Guess length ({len(guess)}) must match feedback length ({len(feedback_string)})")
    if not all(c in 'GYB' for c in feedback_string):
        raise ValueError("Feedback string must only contain G/Y/B characters")

    colored_letters = []
    for letter, feedback in zip(guess, feedback_string):
        if feedback == 'G':
            colored_letters.append(f"\033[92m{letter}\033[0m")  # Green
        elif feedback == 'Y':
            colored_letters.append(f"\033[93m{letter}\033[0m")  # Yellow
        else:  # 'B'
            colored_letters.append(f"\033[90m{letter}\033[0m")  # Gray

    return ' '.join(colored_letters)

def print_known_letters(feedback_matrix):
    """
    Print known letter information from feedback matrix.
    
    Args:
        feedback_matrix (numpy array): The feedback matrix from the environment
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    green_letters = []
    yellow_letters = []
    gray_letters = []
    
    for i in range(26):  # For each letter
        for pos in range(5):  # For each position
            if feedback_matrix[pos, i, 0] > 0:  # Green
                green_letters.append(f"{alphabet[i]} at position {pos+1}")
            elif feedback_matrix[pos, i, 1] > 0:  # Yellow
                yellow_letters.append(alphabet[i])
            elif feedback_matrix[pos, i, 2] > 0:  # Gray
                gray_letters.append(alphabet[i])
    
    print("\nKnown information:")
    if green_letters:
        print("  Correct positions:", ', '.join(green_letters))
    if yellow_letters:
        print("  Correct letters:", ', '.join(set(yellow_letters)))
    if gray_letters:
        print("  Incorrect letters:", ', '.join(set(gray_letters)))