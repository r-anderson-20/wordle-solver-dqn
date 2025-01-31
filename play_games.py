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
from utils import flatten_state, colorize_feedback, print_known_letters, load_words
from collections import Counter
from collections import defaultdict

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
        try:
            next_feedback_matrix, next_valid_mask, next_remaining_guesses, reward, done = env.step(guess)
        except ValueError as e:
            print(f"\nError: {e}")
            # Record the guess and feedback before returning
            guesses.append(guess)
            feedbacks.append(env._compute_feedback_string(guess, secret_word))
            return False, num_guesses, guesses, feedbacks
        
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

def main(words=None):
    """
    Main function to play multiple games and show statistics.
    
    Args:
        words (list[str], optional): List of words to use for both valid guesses and secret words.
                                   If None, loads test words from file.
    """
    # If no words provided, load test words
    if words is None:
        words = load_words('word_lists/test_words.txt')
        
    print(f"Using {len(words)} words for gameplay")
    
    # Create environment and agent
    env = WordleEnvironment(valid_words=words, max_guesses=6)
    state_dim = 390 + len(words) + 1
    action_dim = len(words)
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
    checkpoint = torch.load("model/dqn_model_final.pth", map_location="cpu")
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
        secret_word = random.choice(words)
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
    # Print numeric guesses first
    for guesses, count in sorted((k, v) for k, v in guess_distribution.items() if isinstance(k, int)):
        print(f"{guesses} guesses: {count} games ({count/num_games*100:.1f}%)")
    # Print failed games last
    if 'X' in guess_distribution:
        count = guess_distribution['X']
        print(f"Failed: {count} games ({count/num_games*100:.1f}%)")
            
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
