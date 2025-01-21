"""
Training module for the Wordle DQN agent.
Implements the training loop, experience replay, and evaluation procedures
for the deep Q-learning agent. Includes functionality for periodic evaluation
and model checkpointing.
"""

import random
import numpy as np
import torch
from environment import WordleEnvironment
from agent import DQNAgent
from replay_buffer import ReplayBuffer
from tqdm import tqdm

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

def train(
    valid_words,
    test_words,
    num_episodes=10000,
    hidden_dim=256,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    target_update_freq=100,
    eval_freq=100,
    eval_episodes=100,
    device="cpu"
):
    """
    Train a DQN agent to play Wordle using deep Q-learning.
    
    Args:
        valid_words (list[str]): List of valid words for training
        test_words (list[str]): List of words to use for evaluation
        num_episodes (int): Total number of training episodes
        hidden_dim (int): Size of hidden layers in Q-network
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor for future rewards
        epsilon_start (float): Initial exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (float): Decay rate for epsilon
        batch_size (int): Size of training batches
        target_update_freq (int): Steps between target network updates
        eval_freq (int): Episodes between evaluations
        eval_episodes (int): Number of episodes for each evaluation
        device (str): Device to use for training ("cpu" or "cuda")
        
    Returns:
        DQNAgent: The trained agent
        
    Notes:
        - Uses experience replay for stable learning
        - Implements epsilon-greedy exploration
        - Periodically evaluates on test set
        - Saves best model based on evaluation performance
    """
    # Initialize training environment with only training words
    env = WordleEnvironment(valid_words=valid_words, max_guesses=6)
    
    # State dimension includes feedback matrix (390), valid mask for valid words, and remaining guesses (1)
    state_dim = 390 + len(valid_words) + 1
    action_dim = len(valid_words)  # Actions are indices into valid_words
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        device=device
    )
    
    # Initialize replay buffer with correct state dimension
    replay_buffer = ReplayBuffer(max_size=10000, state_dim=state_dim)
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    solved_episodes = 0
    eval_solve_rates = []
    running_avg_reward = []
    running_avg_length = []
    best_solved_rate = 0
    
    # Progress tracking
    progress_bar = tqdm(range(num_episodes), desc="Training Progress")
    running_solve_rate = 0
    running_reward = 0
    
    print("\nStarting training...")
    print("Episode | Solved | Avg Reward | Epsilon")
    print("-" * 45)
    
    # Training loop
    for episode in progress_bar:
        secret_word = random.choice(valid_words)  # Only train on training words
        feedback_matrix, valid_mask, remaining_guesses = env.reset(secret_word)
        
        state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
        
        episode_reward = 0
        episode_length = 0
        solved = False
        
        # Episode loop
        while True:
            # Select action using valid mask
            action = agent.select_action(state, valid_mask)
            guess = valid_words[action]
            
            # Take step
            feedback_matrix, valid_mask, remaining_guesses, reward, done = env.step(guess)
            state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
            
            # Store transition
            replay_buffer.add(state, action, reward, state, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Train if we have enough samples
            if replay_buffer.size >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.learn(batch)
            
            if done:
                if guess == secret_word:
                    solved_episodes += 1
                    solved = True
                break
            
        # Update metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update running averages
        window = min(100, len(episode_rewards))
        avg_reward = np.mean(episode_rewards[-window:])
        avg_length = np.mean(episode_lengths[-window:])
        running_avg_reward.append(avg_reward)
        running_avg_length.append(avg_length)
        
        # Update progress tracking
        running_solve_rate = solved_episodes / (episode + 1) * 100
        running_reward = avg_reward
        
        # Update progress bar
        progress_bar.set_postfix({
            'Solve Rate': f'{running_solve_rate:.1f}%',
            'Avg Reward': f'{running_reward:.1f}',
            'Epsilon': f'{agent.epsilon:.3f}'
        })
        
        # Print detailed stats periodically
        if (episode + 1) % 100 == 0:
            print(f"{episode+1:7d} | {running_solve_rate:6.1f}% | {running_reward:10.1f} | {agent.epsilon:.3f}")
        
        # Evaluate periodically
        if (episode + 1) % eval_freq == 0:
            print("\nEvaluating...")
            original_epsilon = agent.epsilon
            agent.epsilon = 0.01  # Small epsilon for evaluation
            
            # Create separate evaluation environment with test words only
            eval_env = WordleEnvironment(valid_words=test_words, max_guesses=6)
            
            eval_solved = 0
            eval_total_guesses = 0
            for eval_episode in range(eval_episodes):
                eval_word = random.choice(test_words)
                print(f"\nEvaluation Episode {eval_episode + 1}")
                print(f"Target word: {eval_word}")
                
                feedback_matrix, valid_mask, remaining_guesses = eval_env.reset(eval_word)
                eval_state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
                
                guesses = []
                episode_guesses = 0
                while True:
                    eval_action = agent.select_action(eval_state, valid_mask)
                    eval_guess = test_words[eval_action]
                    episode_guesses += 1
                    
                    guesses.append(eval_guess)
                    print(f"Guess {6-remaining_guesses+1}: {eval_guess}")
                    print(f"Valid words remaining: {sum(valid_mask)}")
                    
                    feedback_matrix, valid_mask, remaining_guesses, reward, done = eval_env.step(eval_guess)
                    eval_state = flatten_state(feedback_matrix, valid_mask, remaining_guesses)
                    
                    if done:
                        if eval_guess == eval_word:
                            eval_solved += 1
                            eval_total_guesses += episode_guesses
                            print(f"Solved in {episode_guesses} guesses!")
                        else:
                            print(f"Failed! Target was {eval_word}. Guesses: {', '.join(guesses)}")
                        break
            
            eval_solve_rate = eval_solved / eval_episodes
            avg_guesses = eval_total_guesses / eval_solved if eval_solved > 0 else 6.0
            eval_solve_rates.append(eval_solve_rate)
            
            print(f"\nEvaluation Results:")
            print(f"Solve Rate: {eval_solve_rate*100:.1f}%")
            print(f"Average Guesses (when solved): {avg_guesses:.2f}")
            
            # Save best model
            if eval_solve_rate > best_solved_rate:
                best_solved_rate = eval_solve_rate
                agent.save("dqn_model_best.pth")
                print(f"New best model saved! Solve rate: {best_solved_rate*100:.1f}%")
            
            agent.epsilon = original_epsilon
    
    print("\nTraining complete!")
    print(f"Final solve rate: {(solved_episodes/num_episodes)*100:.1f}%")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Best evaluation solve rate: {best_solved_rate*100:.1f}%")
    
    # Save final model
    agent.save("dqn_model_final.pth")
    print("Final model saved to dqn_model_final.pth")
    
    return agent, {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'running_avg_reward': running_avg_reward,
        'running_avg_length': running_avg_length,
        'eval_solve_rates': eval_solve_rates,
        'best_solved_rate': best_solved_rate
    }

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

if __name__ == "__main__":
    # Load words
    train_words = load_words('data/train_words.txt')
    test_words = load_words('data/test_words.txt')
    
    print(f"Loaded {len(train_words)} training words and {len(test_words)} test words")
    
    # Train the agent
    trained_agent, metrics = train(
        valid_words=train_words,
        test_words=test_words,
        num_episodes=1000,
        hidden_dim=256,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100,
        eval_freq=200,
        eval_episodes=50,
        device="cpu"
    )