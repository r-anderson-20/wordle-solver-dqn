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
from train import flatten_state
from collections import Counter
from collections import defaultdict

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
    num_guesses = 0
    
    while True:
        print_known_letters(feedback_matrix)
        print(f"Remaining guesses: {remaining_guesses}")
        
        # Get agent's action
        state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
        action = agent.select_action(state, valid_mask)
        guess = env.valid_words[action]
        num_guesses += 1
        
        # Take step in environment
        next_feedback_matrix, next_valid_mask, next_remaining_guesses, reward, done = env.step(guess)
        
        # Get feedback string for visualization
        feedback_string = env._compute_feedback_string(guess, secret_word)
        guesses.append(guess)
        feedbacks.append(feedback_string)
        
        # Print the guess and feedback
        print(f"\nGuess {num_guesses}: {colorize_feedback(guess, feedback_string)}")
        
        if done:
            solved = (guess == secret_word)
            if solved:
                print(f"\n✨ Solved in {num_guesses} guesses!")
            else:
                print(f"\n❌ Failed to solve. The word was: {secret_word}")
            break
            
        feedback_matrix = next_feedback_matrix
        valid_mask = next_valid_mask
        remaining_guesses = next_remaining_guesses
    
    return solved, num_guesses, guesses, feedbacks

def main():
    """
    Main function to play multiple games and show statistics.
    """
    # Load words
    test_words = []
    with open('data/test_words.txt', 'r') as f:
        test_words = [line.strip() for line in f]
    print(f"Loaded {len(test_words)} test words")
    
    # Create environment and agent
    env = WordleEnvironment(valid_words=test_words, max_guesses=6)
    state_dim = 390 + len(test_words) + 1
    action_dim = len(test_words)
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=0.01,
        epsilon_end=0.01,
        epsilon_decay=1.0,
        target_update_freq=1000,
        device="cpu"
    )
    
    # Load trained model
    checkpoint = torch.load("dqn_model_final.pth", map_location="cpu")
    agent.online_net.load_state_dict(checkpoint['online_net'])
    agent.online_net.eval()
    
    # Play games
    num_games = 20
    print(f"\nPlaying {num_games} games using test words...")
    print("-" * 40)
    
    solved_count = 0
    total_guesses = 0
    guess_distribution = defaultdict(int)
    letter_accuracy = defaultdict(list)
    common_first_guesses = defaultdict(int)
    
    for _ in range(num_games):
        secret_word = random.choice(test_words)
        solved, num_guesses, guesses, feedbacks = play_game(env, agent, secret_word)
        
        if solved:
            solved_count += 1
            total_guesses += num_guesses
            guess_distribution[num_guesses] += 1
        else:
            guess_distribution['X'] += 1
            
        # Track first guesses
        if guesses:
            common_first_guesses[guesses[0]] += 1
            
        # Track letter accuracy
        for guess in guesses:
            for pos, (guess_letter, true_letter) in enumerate(zip(guess, secret_word)):
                letter_accuracy[pos].append(guess_letter == true_letter)
    
    # Print statistics
    print("\nGame Statistics:")
    print(f"Solved: {solved_count}/{num_games} ({solved_count/num_games*100:.1f}%)")
    if solved_count > 0:
        print(f"Average guesses when solved: {total_guesses/solved_count:.2f}")
    
    print("\nGuess Distribution:")
    for guesses, count in sorted(guess_distribution.items()):
        if guesses == 'X':
            print(f"Failed: {count} games")
        else:
            print(f"{guesses} guesses: {count} games ({count/num_games*100:.1f}%)")
            
    print("\nLetter Accuracy by Position:")
    for pos, accuracies in letter_accuracy.items():
        avg_accuracy = sum(accuracies) / len(accuracies) * 100
        print(f"Position {pos+1}: {avg_accuracy:.1f}%")
    
    print("\nFirst Guess Analysis:")
    print("Most common first guesses:")
    top_guesses = sorted(common_first_guesses.items(), key=lambda x: x[1], reverse=True)
    for guess, count in top_guesses:
        print(f"  {guess}: {count} times ({count/num_games*100:.1f}%)")

if __name__ == "__main__":
    main()
