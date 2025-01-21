"""
Interactive script for playing Wordle with the trained DQN agent.
Allows visualization of the agent's decision-making process and provides
colored feedback for guesses.
"""

import random
import numpy as np
import torch
from environment import WordleEnvironment
from agent import DQNAgent
from train_minimal import flatten_state

def colorize_feedback(guess, feedback_string):
    """
    Convert feedback string to colored output.
    
    Args:
        guess (str): The guessed word
        feedback_string (str): The feedback string from the environment
        
    Returns:
        str: The colored feedback string
    """
    """Convert feedback string to colored output."""
    result = []
    for letter, fb in zip(guess, feedback_string):
        if fb == 'G':
            result.append(f'\033[92m{letter}\033[0m')  # Green
        elif fb == 'Y':
            result.append(f'\033[93m{letter}\033[0m')  # Yellow
        else:
            result.append(f'\033[90m{letter}\033[0m')  # Gray
    return ' '.join(result)

def print_known_letters(feedback_matrix):
    """
    Print known letter information from feedback matrix.
    
    Args:
        feedback_matrix (numpy array): The feedback matrix from the environment
    """
    """Print known letter information from feedback matrix."""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    green_letters = []
    yellow_letters = []
    gray_letters = []
    
    for i in range(26):  # For each letter
        for pos in range(5):  # For each position
            if feedback_matrix[pos, i, 0] > 0:  # Green
                green_letters.append(f"{alphabet[i].upper()} at position {pos+1}")
            elif feedback_matrix[pos, i, 1] > 0:  # Yellow
                yellow_letters.append(alphabet[i].upper())
            elif feedback_matrix[pos, i, 2] > 0:  # Gray
                gray_letters.append(alphabet[i].upper())
    
    print("Known information:")
    if green_letters:
        print("  Correct positions:", ', '.join(green_letters))
    if yellow_letters:
        print("  Correct letters:", ', '.join(set(yellow_letters)))
    if gray_letters:
        print("  Incorrect letters:", ', '.join(set(gray_letters)))

def play_game(env, agent, secret_word):
    """
    Play a single game of Wordle with visualization.
    
    Args:
        env (WordleEnvironment): The Wordle environment instance
        agent (DQNAgent): The trained DQN agent
        secret_word (str): The target word to guess
        
    Returns:
        tuple: (solved, num_guesses, guesses, feedbacks) - Whether the word was solved, number of guesses used, guesses, and feedbacks
    """
    """Play a single game and return the number of guesses and if solved."""
    feedback_matrix, valid_mask, remaining_guesses = env.reset(secret_word)
    print(f"\nTarget word: {secret_word}")
    print("-" * 40)
    
    guesses = []
    feedbacks = []
    solved = False
    
    while True:
        print_known_letters(feedback_matrix)
        print(f"Remaining guesses: {remaining_guesses}")
        
        # Get agent's action
        state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
        action = agent.select_action(state, valid_mask)
        guess = env.valid_words[action]
        
        # Take step in environment
        next_feedback_matrix, next_valid_mask, next_remaining_guesses, reward, done = env.step(guess)
        
        # Get feedback string for visualization
        feedback_string = env._compute_feedback_string(guess, secret_word)
        guesses.append(guess)
        feedbacks.append(feedback_string)
        
        # Print the guess and feedback
        print(f"\nGuess {6-remaining_guesses}: {colorize_feedback(guess, feedback_string)}")
        
        if done:
            solved = (guess == secret_word)
            if solved:
                print(f"\n✨ Solved in {6-remaining_guesses} guesses!")
            else:
                print(f"\n❌ Failed to solve. The word was: {secret_word}")
            break
            
        feedback_matrix = next_feedback_matrix
        valid_mask = next_valid_mask
        remaining_guesses = next_remaining_guesses
    
    return solved, 6-remaining_guesses, guesses, feedbacks

def main():
    """
    Main function to set up the environment and agent for interactive gameplay.
    Loads the trained model and allows playing multiple games while visualizing
    the agent's decision process.
    """
    # Load test words (using the same set as training)
    with open('data/train_words.txt', 'r') as f:
        all_words = [line.strip() for line in f if line.strip()][:50]  # Use only first 50 words
    test_words = random.sample(all_words, 10)  # Randomly sample 10 words to test
    
    # Initialize environment and agent
    env = WordleEnvironment(valid_words=all_words, max_guesses=6)
    state_dim = 390 + len(all_words) + 1  # feedback_matrix + valid_mask + remaining_guesses
    action_dim = len(all_words)
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        device="cpu"
    )
    
    # Load the trained model
    agent.load("dqn_model_minimal.pth")
    
    # Play 10 games
    total_games = 10
    solved_games = 0
    total_guesses = 0
    
    print(f"\nPlaying {total_games} games...")
    print("=" * 50)
    
    for i in range(total_games):
        secret_word = test_words[i]
        solved, num_guesses, guesses, feedbacks = play_game(env, agent, secret_word)
        
        if solved:
            solved_games += 1
            total_guesses += num_guesses
    
    # Print statistics
    print("\nFinal Statistics:")
    print(f"Games played: {total_games}")
    print(f"Games solved: {solved_games}")
    print(f"Success rate: {(solved_games/total_games)*100:.1f}%")
    if solved_games > 0:
        print(f"Average guesses when solved: {total_guesses/solved_games:.1f}")

if __name__ == "__main__":
    main()
