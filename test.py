"""
Testing module for the Wordle DQN agent.
Implements evaluation procedures for assessing the agent's performance
on unseen test words. Provides detailed statistics on solve rate,
average guesses needed, and distribution of guess counts.
"""

import random
import numpy as np
import torch
from environment import WordleEnvironment
from agent import DQNAgent
from train import flatten_state
from collections import defaultdict

def test(
    train_words,  # Words the agent can use as guesses
    test_words,   # Words to test on
    model_path="dqn_model.pth",
    hidden_dim=256,
    device="cpu"
):
    """
    Evaluate a trained DQNAgent on a set of unseen words.

    Args:
        train_words (list[str]): List of valid words the agent can use as guesses
        test_words (list[str]): List of words to test the agent on
        model_path (str): Path to the saved model file
        hidden_dim (int): Hidden dimension of the QNetwork (must match training)
        device (str): "cpu" or "cuda" for running the model inference

    Returns:
        tuple: (success_rate, avg_guesses, metrics) where:
            - success_rate (float): Percentage of words solved
            - avg_guesses (float): Average number of guesses for solved words
            - metrics (dict): Additional metrics including:
                - guess_distribution: Dictionary mapping number of guesses to count
                - num_solved: Total number of words solved
                - total_words: Total number of words attempted
    """
    # Create environment with training words as valid guesses
    env = WordleEnvironment(valid_words=train_words, max_guesses=6)

    # Initialize agent
    state_dim = 390 + len(train_words) + 1
    action_dim = len(train_words)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=0.01,  # Small epsilon for some exploration
        epsilon_end=0.01,
        epsilon_decay=1.0,
        target_update_freq=1000,
        device=device
    )

    # Load trained weights
    agent.online_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.online_net.eval()  # Set to evaluation mode

    # Evaluation metrics
    total_words = len(test_words)
    solved_words = 0
    total_guesses = 0
    guess_distribution = defaultdict(int)  # Track number of guesses needed
    letter_accuracy = defaultdict(list)    # Track accuracy by position
    common_first_guesses = defaultdict(int)  # Track most common first guesses
    
    print("\nStarting evaluation...")
    
    for i, test_word in enumerate(test_words, 1):
        if i % 50 == 0:
            print(f"Testing word {i}/{total_words}...")
            
        feedback_matrix, valid_mask, remaining_guesses = env.reset(test_word)
        state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
        
        guesses = []
        solved = False
        
        while True:
            action = agent.select_action(state, valid_mask, eval_mode=True)
            guess = train_words[action]
            guesses.append(guess)
            
            if len(guesses) == 1:
                common_first_guesses[guess] += 1
            
            if guess == test_word:
                solved = True
                solved_words += 1
                num_guesses = len(guesses)
                total_guesses += num_guesses
                guess_distribution[num_guesses] += 1
                break
                
            if env.done:
                guess_distribution['X'] += 1  # Mark unsolved puzzles
                break
                
            next_feedback_matrix, next_valid_mask, next_remaining_guesses = env.step(guess)
            state = flatten_state(next_feedback_matrix, next_valid_mask, next_remaining_guesses)
            
            # Record letter accuracy
            for pos, (guess_letter, true_letter) in enumerate(zip(guess, test_word)):
                letter_accuracy[pos].append(guess_letter == true_letter)
    
    # Calculate metrics
    success_rate = (solved_words / total_words) * 100
    avg_guesses = total_guesses / solved_words if solved_words > 0 else 6.0
    
    # Print detailed results
    print("\nDetailed Evaluation Results:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Guesses (when solved): {avg_guesses:.2f}")
    
    print("\nGuess Distribution:")
    for guesses, count in sorted(guess_distribution.items()):
        if guesses == 'X':
            print(f"Failed: {count} words")
        else:
            print(f"{guesses} guesses: {count} words ({count/total_words*100:.1f}%)")
    
    print("\nLetter Accuracy by Position:")
    for pos, accuracies in letter_accuracy.items():
        avg_accuracy = sum(accuracies) / len(accuracies) * 100
        print(f"Position {pos+1}: {avg_accuracy:.1f}%")
    
    print("\nTop 5 First Guesses:")
    top_guesses = sorted(common_first_guesses.items(), key=lambda x: x[1], reverse=True)[:5]
    for guess, count in top_guesses:
        print(f"{guess}: {count} times ({count/total_words*100:.1f}%)")
    
    metrics = {
        "guess_distribution": dict(guess_distribution),
        "num_solved": solved_words,
        "total_words": total_words
    }
    
    return success_rate, avg_guesses, metrics

if __name__ == "__main__":
    # Example usage with proper word lists
    def load_words(filename):
        """
        Load words from a text file, one word per line.
        
        Args:
            filename (str): Path to the word list file
        
        Returns:
            list[str]: List of words from the file
        """
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    train_words = load_words('data/train_words.txt')
    test_words = load_words('data/test_words.txt')
    
    success_rate, avg_guesses, metrics = test(
        train_words=train_words,
        test_words=test_words,
        model_path="dqn_model.pth"
    )