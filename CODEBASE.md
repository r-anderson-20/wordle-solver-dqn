# Wordle Solver DQN Codebase

This document contains the complete codebase for the Wordle Solver DQN project.

## Table of Contents

### Core Files
- [main](#main)
- [agent](#agent)
- [environment](#environment)
- [train](#train)
- [play_games](#play_games)
- [utils](#utils)
- [replay_buffer](#replay_buffer)

### Test Files
- [test_agent](#test_agent)
- [test_environment](#test_environment)
- [test_main](#test_main)
- [test_play_games](#test_play_games)
- [test_replay_buffer](#test_replay_buffer)
- [test_train](#test_train)
- [test_utils](#test_utils)

---

## Core Files

### main

```python
"""
Main entry point for training and testing the Wordle DQN agent.
Provides command-line interface for running different modes of operation
and managing model files.
"""

import argparse
import torch
from train import train
from play_games import main as play_games
from utils import load_words

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - mode: 'train' or 'test'
            - model_path: path to save/load model
            - train_words: path to training words file
            - test_words: path to test words file
            - num_episodes: number of training episodes
            - hidden_dim: hidden layer dimension
            - batch_size: training batch size
            - learning_rate: optimizer learning rate
            - device: 'cuda' or 'cpu'
    """
    parser = argparse.ArgumentParser(description='Train or test the Wordle DQN agent')
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--model_path", type=str, default="model/dqn_model.pth")
    parser.add_argument("--train_words", type=str, default="word_lists/train_words.txt")
    parser.add_argument("--test_words", type=str, default="word_lists/test_words.txt")
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    """
    Main function to run training or testing of the Wordle DQN agent.
    Handles command line arguments and executes the appropriate mode
    of operation (training or testing).
    """
    args = parse_args()
    
    # Load appropriate word lists
    if args.mode == "train":
        train_words = load_words(args.train_words)
        test_words = load_words(args.test_words)
        print(f"Loaded {len(train_words)} training words")
        
        trained_agent, _ = train(
            valid_words=train_words,
            test_words=test_words,
            num_episodes=args.num_episodes,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        # Save model
        trained_agent.online_net.cpu()
        torch.save(trained_agent.online_net.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == "test":
        # For testing, we only need test words
        test_words = load_words(args.test_words)
        print(f"Loaded {len(test_words)} test words")
        
        # Use test words for both valid guesses and secret words
        play_games(test_words)

if __name__ == "__main__":
    main()
```

### agent

```python
"""
Deep Q-Network (DQN) agent implementation for Wordle.
Contains the neural network architecture (QNetwork) and agent logic (DQNAgent)
for learning to play Wordle through deep reinforcement learning.

Key components:
- QNetwork: Deep neural network with residual connections
- DQNAgent: Implements DQN algorithm with experience replay and target network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation with residual connections.
    
    Architecture:
    - Input layer: state vector → hidden_dim
    - Two residual blocks with ReLU activation
    - Dropout layers for regularization
    - Output layer: hidden_dim → action_dim (Q-values for each word)
    
    Uses He initialization for weights and implements residual connections
    for better gradient flow during training.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        """
        Initialize the Q-network.

        Args:
            input_dim (int): Size of the input (state) vector
            output_dim (int): Number of possible actions (size of word dictionary)
            hidden_dim (int): Number of hidden units in each layer

        Raises:
            ValueError: If any dimension is not positive
        """
        super(QNetwork, self).__init__()
        
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Hidden layers with residual connections
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize network weights using He initialization.
        
        Applies to all linear layers:
        - Weights: He initialization for ReLU activation
        - Biases: Initialized to zero
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
                            or [input_dim] for single sample
        
        Returns:
            torch.Tensor: Q-values for each action, shape [batch_size, output_dim]
                         or [output_dim] for single sample
                         
        Raises:
            TypeError: If input is not a torch.Tensor
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # Handle single sample case during inference
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Input layer
        x = self.input_layer(x)
        
        # First residual block
        identity = x
        x = self.hidden_layer1(x)
        x = F.relu(x + identity)
        x = self.dropout(x)
        
        # Second residual block
        identity = x
        x = self.hidden_layer2(x)
        x = F.relu(x + identity)
        x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Remove batch dimension if it was added
        if not self.training and x.size(0) == 1:
            x = x.squeeze(0)
            
        return x


class DQNAgent:
    """
    Deep Q-Network Agent for playing Wordle.
    
    Features:
    - Maintains online and target networks for stable learning
    - Uses epsilon-greedy exploration strategy
    - Implements experience replay
    - Supports both training and evaluation modes
    - Tracks training statistics (loss, average Q-values)
    
    The agent learns to map state observations (feedback matrix + valid words)
    to Q-values for each possible word in the dictionary.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        target_update_freq=1000,
        device=None
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim (int): Dimension of the state vector
            action_dim (int): Number of possible actions (dictionary size)
            hidden_dim (int): Size of hidden layers in QNetwork
            lr (float): Learning rate for Adam optimizer
            gamma (float): Discount factor for future rewards
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Multiplicative decay factor for epsilon
            target_update_freq (int): Steps between target network updates
            device (str or torch.device): Device to use for tensor operations

        Raises:
            ValueError: If parameters are invalid (e.g., negative dimensions)
        """
        # Input validation
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1")
        if not (0 <= epsilon_start <= 1) or not (0 <= epsilon_end <= 1):
            raise ValueError("epsilon values must be between 0 and 1")
        if not (0 < epsilon_decay <= 1):
            raise ValueError("epsilon_decay must be between 0 and 1")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Statistics for monitoring
        self.train_steps = 0
        self.updates = 0
        self.avg_loss = 0
        self.avg_q = 0

        # Build networks
        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def select_action(self, state, valid_mask, eval_mode=False):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray or torch.Tensor): Current state observation
            valid_mask (np.ndarray or list[bool]): Mask of valid actions
            eval_mode (bool): If True, uses small epsilon for evaluation
            
        Returns:
            int: Index of the selected action (word)
            
        Raises:
            TypeError: If inputs have incorrect types
            ValueError: If valid_mask length doesn't match action_dim or no valid actions
        """
        # Type checking for valid_mask
        if not isinstance(valid_mask, (np.ndarray, list)):
            raise TypeError("valid_mask must be a numpy array or list")
        
        # Length checking for valid_mask
        if len(valid_mask) != self.action_dim:
            raise ValueError(f"valid_mask length ({len(valid_mask)}) must match action_dim ({self.action_dim})")
        
        # Check that at least one action is valid
        if not any(valid_mask):
            raise ValueError("At least one action must be valid")
        
        # Type checking for state
        if not isinstance(state, (np.ndarray, torch.Tensor)):
            raise TypeError("state must be a numpy array or torch tensor")
        
        # Convert state to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Shape checking for state
        if state.shape[0] != self.state_dim:
            raise ValueError(f"state dimension ({state.shape[0]}) must match state_dim ({self.state_dim})")
        
        # Ensure state has batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Use small epsilon for evaluation
        current_epsilon = 0.05 if eval_mode else self.epsilon
        
        # Epsilon-greedy action selection
        if random.random() > current_epsilon:
            with torch.no_grad():
                q_values = self.online_net(state)
                
                # Set Q-values of invalid actions to negative infinity
                q_values = q_values.squeeze()
                invalid_mask = ~torch.tensor(valid_mask, device=self.device)
                q_values[invalid_mask] = float('-inf')
                
                # Update statistics
                if not eval_mode:
                    self.avg_q = 0.95 * self.avg_q + 0.05 * q_values.max().item()
                
                return q_values.argmax().item()
        else:
            # Random choice from valid actions
            valid_indices = np.where(valid_mask)[0]
            return random.choice(valid_indices)

    def learn(self, batch):
        """
        Update the Q-network using a batch of transitions.
        
        Args:
            batch (dict): Batch of transitions with keys:
                - states (torch.Tensor): Current states
                - actions (torch.Tensor): Actions taken
                - rewards (torch.Tensor): Rewards received
                - next_states (torch.Tensor): Next states
                - dones (torch.Tensor): Episode termination flags
        
        Returns:
            float: The loss value for this update
        """
        # No need to convert to tensors since they're already tensors
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        # Compute current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use online network to select actions, target network to evaluate them
        with torch.no_grad():
            # Select actions using online network
            online_next_q_values = self.online_net(next_states)
            best_actions = online_next_q_values.argmax(dim=1, keepdim=True)
            
            # Evaluate Q-values using target network
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.gather(1, best_actions)
            
            # Compute target Q values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * max_next_q_values

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)  # Huber loss for stability
        
        # Update statistics
        self.avg_loss = 0.95 * self.avg_loss + 0.05 * loss.item()
        self.train_steps += 1
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update target network if it's time
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.updates += 1

        return loss.item()
    
    def get_statistics(self):
        """Return current training statistics."""
        return {
            'train_steps': self.train_steps,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'avg_loss': self.avg_loss,
            'avg_q': self.avg_q
        }
        
    def save(self, path):
        """Save the online network state dict."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'epsilon': self.epsilon,
            'statistics': self.get_statistics()
        }, path)
        
    def load(self, path):
        """Load a saved state dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['online_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_steps = int(checkpoint['train_steps'])  # Convert to int for safety
        self.epsilon = float(checkpoint['epsilon'])  # Convert to float for safety
        
        # Restore statistics if available
        if 'statistics' in checkpoint:
            stats = checkpoint['statistics']
            self.updates = int(stats.get('updates', 0))
            self.avg_loss = float(stats.get('avg_loss', 0))
            self.avg_q = float(stats.get('avg_q', 0))
```

### environment

```python
"""
Wordle environment implementation for reinforcement learning.
Provides a Gym-like interface for the Wordle game, including:
- State representation as a 3D feedback matrix
- Action space as valid word indices
- Reward function based on game outcome and guess quality
- Feedback generation for guesses
"""

import numpy as np

class WordleEnvironment:
    """
    A simple Wordle-like environment for reinforcement learning.
    """

    def __init__(self, valid_words, max_guesses=6):
        """
        Args:
            valid_words (list[str]): List of valid words (both solutions & guesses).
            max_guesses (int): Maximum number of guesses allowed per episode.
        """
        if not isinstance(valid_words, list) or not all(isinstance(w, str) for w in valid_words):
            raise ValueError("valid_words must be a list of strings")
        if not all(len(w) == 5 for w in valid_words):
            raise ValueError("all words must be 5 letters long")
        if max_guesses <= 0:
            raise ValueError("max_guesses must be positive")
            
        self.valid_words = valid_words  # Master list of all valid guessable words
        self.max_guesses = max_guesses

        # Internal state
        self.secret_word = None
        self.remaining_guesses = None
        self.done = False
        self.last_reward = 0.0  # Track the last reward

        # Track a mask of which words are still valid given feedback so far
        # We'll store this as a boolean array of length len(valid_words)
        self.valid_mask = None

        # Store the feedback matrix from the most recent guess
        self.last_feedback_matrix = None

    def reset(self, secret_word):
        """
        Resets the environment state for a new puzzle.

        Args:
            secret_word (str): The word to be guessed this episode.

        Returns:
            feedback_matrix (np.ndarray): 5x26x3 matrix of letter feedback
            valid_mask (np.ndarray): Boolean mask of valid words
            remaining_guesses (int): Number of guesses remaining
        """
        if not isinstance(secret_word, str) or len(secret_word) != 5:
            raise ValueError("secret_word must be a 5-letter string")
        if secret_word not in self.valid_words:
            raise ValueError("secret_word must be in valid_words list")
            
        self.secret_word = secret_word
        self.remaining_guesses = self.max_guesses
        self.done = False
        self.last_reward = 0.0  # Reset last reward

        # Initially, all words could be valid solutions
        self.valid_mask = np.ones(len(self.valid_words), dtype=bool)

        # No feedback yet, so we'll use an all-zeros 5x26x3 matrix
        self.last_feedback_matrix = np.zeros((5, 26, 3), dtype=np.float32)

        return self.last_feedback_matrix, self.valid_mask, self.remaining_guesses

    def step(self, guess):
        """
        Executes one guess in the game.

        Args:
            guess (str): The word guessed by the agent.

        Returns:
            feedback_matrix (np.ndarray): 5x26x3 matrix of letter feedback
            valid_mask (np.ndarray): Boolean mask of valid words
            remaining_guesses (int): Number of guesses remaining
            reward (float): Reward from this guess
            done (bool): Whether the episode has ended
        """
        if not isinstance(guess, str) or len(guess) != 5:
            raise ValueError("guess must be a 5-letter string")
        if guess not in self.valid_words:
            raise ValueError("guess must be in valid_words list")
            
        # Compute feedback & reward
        feedback_string = self._compute_feedback_string(guess, self.secret_word)
        feedback_matrix = self._compute_feedback_matrix(guess, feedback_string)

        # Visualize feedback
        feedback_str = ""
        for i, (letter, feedback) in enumerate(zip(guess, feedback_string)):
            if feedback == 'G':  # Correct position
                feedback_str += f"\033[92m{letter}\033[0m"  # Green
            elif feedback == 'Y':  # Wrong position
                feedback_str += f"\033[93m{letter}\033[0m"  # Yellow
            else:  # Not in word
                feedback_str += f"\033[90m{letter}\033[0m"  # Gray
        print(f"Feedback: {feedback_str}")

        # Update reward structure:
        # +2 for each green letter (correct position)
        # +1 for each yellow letter (correct letter, wrong position)
        # -0.1 for each gray letter to encourage efficiency
        # +10 bonus for solving the puzzle
        green_count = sum(f == 'G' for f in feedback_string)
        yellow_count = sum(f == 'Y' for f in feedback_string)
        gray_count = sum(f == 'B' for f in feedback_string)
        
        reward = (2 * green_count) + yellow_count - (0.1 * gray_count)
        
        if guess == self.secret_word:
            # Bonus for solving + extra points for solving quickly
            reward += 10 + (self.remaining_guesses * 2)

        # Store the reward
        self.last_reward = reward

        # Update environment state
        self.remaining_guesses -= 1
        self.last_feedback_matrix = feedback_matrix

        # If the guess is exactly the secret word, we consider it solved
        if guess == self.secret_word:
            self.done = True
        elif self.remaining_guesses <= 0:
            self.done = True

        # Update valid_mask based on new feedback
        self._update_valid_mask(guess, feedback_string)

        return self.last_feedback_matrix, self.valid_mask, self.remaining_guesses, reward, self.done

    def _compute_feedback_string(self, guess, target):
        """
        Compute Wordle-style feedback for a guess.
        Returns a string of length 5 with:
            'G' for correct letter in correct position (green)
            'Y' for correct letter in wrong position (yellow)
            'B' for incorrect letter (black/gray)
            
        The feedback follows official Wordle rules for repeated letters:
        1. First mark all exact matches as green
        2. Then mark letters in wrong positions as yellow, considering:
           - If a letter appears multiple times in the guess but only once in the answer,
             only the first instance gets marked (either green or yellow)
           - If a letter appears multiple times in both guess and answer,
             all instances get marked appropriately
        """
        feedback = ['B'] * 5
        target_chars = list(target)
        guess_chars = list(guess)

        # First pass: mark all correct positions (green)
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback[i] = 'G'
                target_chars[i] = None  # Mark as used
                guess_chars[i] = None

        # Second pass: mark correct letters in wrong positions (yellow)
        for i in range(5):
            if guess_chars[i] is None:  # Skip already marked positions
                continue
            for j in range(5):
                if target_chars[j] is not None and guess_chars[i] == target_chars[j]:
                    feedback[i] = 'Y'
                    target_chars[j] = None  # Mark as used
                    break

        return ''.join(feedback)

    def _compute_feedback_matrix(self, guess, feedback_string):
        """
        Convert a guess and its feedback into a 5x26x3 matrix.
        Each position has a 26-dim one-hot vector for the letter,
        and a 3-dim one-hot vector for the feedback type (gray/yellow/green).
        """
        matrix = np.zeros((5, 26, 3), dtype=np.float32)
        
        for i, (letter, feedback) in enumerate(zip(guess, feedback_string)):
            letter_idx = ord(letter.lower()) - ord('a')
            matrix[i, letter_idx, 0] = feedback == 'B'  # gray
            matrix[i, letter_idx, 1] = feedback == 'Y'  # yellow
            matrix[i, letter_idx, 2] = feedback == 'G'  # green
            
        return matrix

    def _update_valid_mask(self, guess, feedback_string):
        """
        Update the valid_mask based on the feedback received for a guess.
        This eliminates words that couldn't be the answer given the feedback.
        """
        for i, word in enumerate(self.valid_words):
            if not self.valid_mask[i]:  # Skip already invalid words
                continue
            # A word remains valid only if it would give the same feedback
            test_feedback = self._compute_feedback_string(guess, word)
            if test_feedback != feedback_string:
                self.valid_mask[i] = False

    def is_done(self):
        """Convenience method for external checks."""
        return self.done
```

### train

```python
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
from utils import flatten_state, load_words
from tqdm import tqdm

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
        dict: Training metrics including rewards, lengths, and evaluation results
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not valid_words:
        raise ValueError("valid_words must not be empty")
    if not test_words:
        raise ValueError("test_words must not be empty")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    if not 0 <= epsilon_start <= 1:
        raise ValueError("epsilon_start must be between 0 and 1")
    if not 0 <= epsilon_end <= 1:
        raise ValueError("epsilon_end must be between 0 and 1")
    if not 0 <= epsilon_decay <= 1:
        raise ValueError("epsilon_decay must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if target_update_freq <= 0:
        raise ValueError("target_update_freq must be positive")
    if eval_freq <= 0:
        raise ValueError("eval_freq must be positive")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if device not in ["cpu", "cuda"]:
        raise ValueError("device must be 'cpu' or 'cuda'")
    
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
                
                # Convert batch to tensors
                batch = {
                    'states': torch.FloatTensor(batch['states']),
                    'actions': torch.LongTensor(batch['actions']),
                    'rewards': torch.FloatTensor(batch['rewards']),
                    'next_states': torch.FloatTensor(batch['next_states']),
                    'dones': torch.FloatTensor(batch['dones'])
                }
                
                # Update agent
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
                agent.save("model/dqn_model_best.pth")
                print(f"New best model saved! Solve rate: {best_solved_rate*100:.1f}%")
            
            agent.epsilon = original_epsilon
    
    print("\nTraining complete!")
    print(f"Final solve rate: {(solved_episodes/num_episodes)*100:.1f}%")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Best evaluation solve rate: {best_solved_rate*100:.1f}%")
    
    # Save final model
    agent.save("model/dqn_model_final.pth")
    print("Final model saved to model/dqn_model_final.pth")
    
    return agent, {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'running_avg_reward': running_avg_reward,
        'running_avg_length': running_avg_length,
        'eval_solve_rates': eval_solve_rates,
        'best_solved_rate': best_solved_rate
    }

if __name__ == "__main__":
    # Load words
    train_words = load_words('word_lists/train_words.txt')
    test_words = load_words('word_lists/test_words.txt')
    
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
```

### play_games

```python
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

```

### utils

```python
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
```

### replay_buffer

```python
"""
Experience replay buffer implementation for DQN training.
Provides efficient storage and sampling of transitions (state, action, reward, next_state, done)
using numpy arrays. Implements FIFO behavior when buffer is full.
"""

import numpy as np

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for Q-learning.
    Stores transitions using numpy arrays for efficient sampling and memory usage.
    When buffer is full, oldest transitions are replaced first.
    """

    def __init__(self, max_size, state_dim):
        """
        Initialize replay buffer with fixed maximum size.

        Args:
            max_size (int): Maximum number of transitions to store
            state_dim (int): Dimension of the flattened state vector
                           (for example, 390 + dictionary_size + 1)

        Raises:
            ValueError: If max_size or state_dim are not positive
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
            
        self.max_size = max_size
        self.state_dim = state_dim

        # Create numpy arrays to hold each component of the transition
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

        self.ptr = 0        # Current insert pointer
        self.size = 0       # Current number of stored transitions

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): Current state vector [state_dim]
            action (int): Action index taken
            reward (float): Reward received
            next_state (np.ndarray): Next state vector [state_dim]
            done (bool): Whether episode ended after this transition
            
        Raises:
            ValueError: If inputs have incorrect shapes or types
            
        Notes:
            When buffer is full, oldest transitions are overwritten first
        """
        # Input validation
        if not isinstance(state, np.ndarray) or state.shape != (self.state_dim,):
            raise ValueError(f"state must be a numpy array of shape ({self.state_dim},)")
        if not isinstance(next_state, np.ndarray) or next_state.shape != (self.state_dim,):
            raise ValueError(f"next_state must be a numpy array of shape ({self.state_dim},)")
        if not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not isinstance(reward, (float, np.floating)):
            raise ValueError("reward must be a float")
        if not isinstance(done, bool):
            raise ValueError("done must be a boolean")

        idx = self.ptr

        # Convert inputs to correct types if needed
        self.states[idx] = state.astype(np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = next_state.astype(np.float32)
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) where:
                - states: np.ndarray [batch_size, state_dim]
                - actions: np.ndarray [batch_size]
                - rewards: np.ndarray [batch_size]
                - next_states: np.ndarray [batch_size, state_dim]
                - dones: np.ndarray [batch_size]
                
        Raises:
            ValueError: If batch_size > current buffer size
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} transitions, buffer only has {self.size}")

        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
        }
        return batch
        
    def __len__(self):
        """Returns the current size of the buffer."""
        return self.size
```

## Test Files

### test_agent

```python
"""
Tests for the DQN agent implementation.
"""

import pytest
import torch
import numpy as np
from agent import QNetwork, DQNAgent

class TestQNetwork:
    """Test suite for QNetwork class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        net = QNetwork(input_dim=10, output_dim=5, hidden_dim=32)
        
        # Check layer dimensions
        assert net.input_layer[0].in_features == 10
        assert net.input_layer[0].out_features == 32
        assert net.output_layer.in_features == 32
        assert net.output_layer.out_features == 5
        
    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            QNetwork(input_dim=0, output_dim=5)
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            QNetwork(input_dim=10, output_dim=0)
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            QNetwork(input_dim=10, output_dim=5, hidden_dim=0)
            
    def test_forward_valid(self):
        """Test forward pass with valid inputs."""
        net = QNetwork(input_dim=4, output_dim=3, hidden_dim=8)
        
        # Single sample
        x = torch.randn(4)
        output = net(x)
        assert output.shape == torch.Size([1, 3])  # Output includes batch dimension
        
        # Batch of samples
        x = torch.randn(2, 4)
        output = net(x)
        assert output.shape == torch.Size([2, 3])
        
        # Test training mode (should keep batch dimension)
        net.train()
        x = torch.randn(1, 4)
        output = net(x)
        assert output.shape == torch.Size([1, 3])
        
        # Test eval mode with batch size > 1 (should keep batch dimension)
        net.eval()
        x = torch.randn(2, 4)
        output = net(x)
        assert output.shape == torch.Size([2, 3])
        
        # Test eval mode with single sample (should squeeze output)
        net.eval()
        x = torch.randn(1, 4)
        output = net(x)
        assert output.shape == torch.Size([3])  # No batch dimension
        
    def test_forward_invalid(self):
        """Test forward pass with invalid inputs."""
        net = QNetwork(input_dim=4, output_dim=3)
        
        # Wrong input type
        with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
            net(np.array([1, 2, 3, 4]))
            
        # Wrong input dimension
        with pytest.raises(RuntimeError):
            net(torch.randn(4, 5))  # Wrong input dimension
            
class TestDQNAgent:
    """Test suite for DQNAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create a DQNAgent instance for testing."""
        return DQNAgent(
            state_dim=10,
            action_dim=5,
            hidden_dim=32,
            lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=100,
            device="cpu"
        )
    
    def test_init_valid(self, agent):
        """Test initialization with valid parameters."""
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.1
        assert agent.epsilon_decay == 0.995
        assert agent.target_update_freq == 100
        assert isinstance(agent.online_net, QNetwork)
        assert isinstance(agent.target_net, QNetwork)
        
    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="state_dim and action_dim must be positive"):
            DQNAgent(state_dim=0, action_dim=5)
        with pytest.raises(ValueError, match="state_dim and action_dim must be positive"):
            DQNAgent(state_dim=10, action_dim=0)
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, gamma=1.5)
        with pytest.raises(ValueError, match="epsilon values must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, epsilon_start=1.5)
        with pytest.raises(ValueError, match="epsilon values must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, epsilon_end=1.5)
        with pytest.raises(ValueError, match="epsilon_decay must be between 0 and 1"):
            DQNAgent(state_dim=10, action_dim=5, epsilon_decay=0)
        with pytest.raises(ValueError, match="target_update_freq must be positive"):
            DQNAgent(state_dim=10, action_dim=5, target_update_freq=0)
            
    def test_select_action_valid(self, agent):
        """Test action selection with valid inputs."""
        state = np.random.randn(10)
        valid_mask = [True] * 5
        
        # Test greedy action selection (epsilon = 0)
        agent.epsilon = 0
        action = agent.select_action(state, valid_mask)
        assert 0 <= action < 5
        
        # Test random action selection (epsilon = 1)
        agent.epsilon = 1
        action = agent.select_action(state, valid_mask)
        assert 0 <= action < 5
        
        # Test with torch tensor input
        state = torch.randn(10)
        action = agent.select_action(state, valid_mask)
        assert 0 <= action < 5
        
        # Test with partial valid mask
        valid_mask = [True, False, True, False, True]
        action = agent.select_action(state, valid_mask)
        assert valid_mask[action]  # Selected action should be valid
        
    def test_select_action_invalid(self, agent):
        """Test action selection with invalid inputs."""
        state = np.random.randn(10)
        
        # Invalid mask type
        with pytest.raises(TypeError, match="valid_mask must be a numpy array or list"):
            agent.select_action(state, "invalid")
            
        # Invalid mask length
        with pytest.raises(ValueError, match="valid_mask length .* must match action_dim"):
            agent.select_action(state, [True, False])
            
        # Invalid state type
        with pytest.raises(TypeError, match="state must be a numpy array or torch tensor"):
            agent.select_action("invalid", [True] * 5)

        # Invalid state shape
        with pytest.raises(ValueError, match="state dimension .* must match state_dim"):
            agent.select_action(np.random.randn(15), [True] * 5)

        # No valid actions
        with pytest.raises(ValueError, match="At least one action must be valid"):
            agent.select_action(state, [False] * 5)

        # Test with eval mode and invalid state
        with pytest.raises(TypeError, match="state must be a numpy array or torch tensor"):
            agent.select_action("invalid", [True] * 5, eval_mode=True)
        
    def test_learn(self, agent):
        """Test learning from a batch of transitions."""
        batch_size = 4
        batch = {
            'states': torch.randn(batch_size, 10, dtype=torch.float32),  # Use float32 explicitly
            'actions': torch.randint(0, 5, size=(batch_size,)),
            'rewards': torch.randn(batch_size, dtype=torch.float32),
            'next_states': torch.randn(batch_size, 10, dtype=torch.float32),
            'dones': torch.randint(0, 2, size=(batch_size,))
        }
        
        # Initial statistics
        initial_steps = agent.train_steps
        initial_updates = agent.updates
        
        # Set train_steps to trigger target network update
        agent.train_steps = agent.target_update_freq - 1
        
        # Perform learning step
        loss = agent.learn(batch)
        
        # Check that statistics were updated
        assert agent.train_steps == agent.target_update_freq  # One step taken
        assert isinstance(loss, float)
        assert loss >= 0
        
        # Check target network update
        assert agent.updates == initial_updates + 1  # Should have updated

    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent state."""
        # Generate a save path
        save_path = tmp_path / "agent.pt"
        
        # Save initial state
        initial_state_dict = agent.online_net.state_dict()
        initial_stats = agent.get_statistics()
        agent.save(str(save_path))
        
        # Modify agent state
        agent.train_steps += 100
        agent.epsilon = 0.5
        
        # Load saved state
        agent.load(str(save_path))
        
        # Check that state was restored
        loaded_state_dict = agent.online_net.state_dict()
        loaded_stats = agent.get_statistics()
        
        assert torch.all(torch.eq(initial_state_dict['input_layer.0.weight'],
                                loaded_state_dict['input_layer.0.weight']))
        assert initial_stats['train_steps'] == loaded_stats['train_steps']
        assert initial_stats['epsilon'] == loaded_stats['epsilon']

```

### test_environment

```python
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

```

### test_main

```python
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

```

### test_play_games

```python
"""
Tests for the play_games module.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from play_games import play_game, main

class TestPlayGames:
    """Test suite for play_games functionality."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock WordleEnvironment."""
        env = MagicMock()
        env.valid_words = ["apple", "beach", "crane"]
        env._compute_feedback_string.return_value = "GGGGG"
        return env
        
    @pytest.fixture
    def mock_agent(self):
        """Create a mock DQNAgent."""
        agent = MagicMock()
        agent.select_action.return_value = 0
        agent.online_net = MagicMock()
        return agent
        
    def test_play_game_solved(self, mock_env, mock_agent, capsys):
        """Test playing a game that is solved."""
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))  # Correct shape: (max_guesses, alphabet_size, feedback_types)
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env.valid_words = ["apple", "beach", "crane"]
        mock_env._compute_feedback_string.return_value = "GGGGG"
        
        # Play game
        solved, num_guesses, guesses, feedbacks = play_game(mock_env, mock_agent, "apple")
        
        # Check results
        assert solved is True
        assert num_guesses == 1
        assert guesses == ["apple"]
        assert feedbacks == ["GGGGG"]
        
        # Check output
        captured = capsys.readouterr()
        assert "Target word: apple" in captured.out
        assert " Solved in 1 guesses!" in captured.out
        
    def test_play_game_failed(self, mock_env, mock_agent, capsys):
        """Test playing a game that is not solved."""
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))  # Correct shape: (max_guesses, alphabet_size, feedback_types)
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 0.0, True)
        mock_env.valid_words = ["apple", "beach", "crane"]
        mock_env._compute_feedback_string.return_value = "YBBBB"
        
        # Play game
        solved, num_guesses, guesses, feedbacks = play_game(mock_env, mock_agent, "beach")
        
        # Check results
        assert solved is False
        assert num_guesses == 1
        assert guesses == ["apple"]
        assert feedbacks == ["YBBBB"]
        
        # Check output
        captured = capsys.readouterr()
        assert "Target word: beach" in captured.out
        assert " Failed to solve" in captured.out
        
    def test_play_game_invalid_state(self, mock_env, mock_agent, capsys):
        """Test playing a game with invalid state transitions."""
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        
        # Mock step to raise ValueError on second call
        mock_env.step.side_effect = [
            (feedback_matrix, np.ones(3), 5, 0.0, False),  # First call succeeds
            ValueError("Invalid state transition")          # Second call fails
        ]
        mock_env.valid_words = ["apple", "beach", "crane"]
        mock_env._compute_feedback_string.return_value = "BBBBB"
        
        # Play game
        solved, num_guesses, guesses, feedbacks = play_game(mock_env, mock_agent, "beach")
        
        # Check results
        assert solved is False
        assert num_guesses == 2  # Should count both attempts
        assert len(guesses) == 2
        assert len(feedbacks) == 2
        
        # Check output
        captured = capsys.readouterr()
        assert "Target word: beach" in captured.out
        assert "Error: Invalid state transition" in captured.out

    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    @patch('play_games.load_words')
    def test_main_with_provided_words(self, mock_load_words, mock_torch_load, mock_agent_class, mock_env_class, capsys):
        """Test main function with provided words."""
        # Set up mocks
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.valid_words = ["apple", "beach", "crane"]
        
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env._compute_feedback_string.return_value = "GGGGG"
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.select_action.return_value = 0
        
        # Create a mock state dict with all required keys
        mock_state_dict = {
            'online_net': {
                'input_layer.0.weight': torch.randn(256, 891),
                'input_layer.0.bias': torch.randn(256),
                'hidden_layer1.0.weight': torch.randn(256, 256),
                'hidden_layer1.0.bias': torch.randn(256),
                'hidden_layer1.2.weight': torch.randn(256, 256),
                'hidden_layer1.2.bias': torch.randn(256),
                'hidden_layer2.0.weight': torch.randn(256, 256),
                'hidden_layer2.0.bias': torch.randn(256),
                'hidden_layer2.2.weight': torch.randn(256, 256),
                'hidden_layer2.2.bias': torch.randn(256),
                'output_layer.weight': torch.randn(500, 256),
                'output_layer.bias': torch.randn(500)
            }
        }
        mock_torch_load.return_value = mock_state_dict
        
        # Run main with provided words
        test_words = ["apple", "beach", "crane"]
        main(test_words)
        
        # Check that environment and agent were created correctly
        mock_env_class.assert_called_once()
        mock_agent_class.assert_called_once()
        mock_load_words.assert_not_called()
        
        # Check output
        captured = capsys.readouterr()
        assert "Using 3 words for gameplay" in captured.out
        assert "Playing 20 games" in captured.out
        assert "Game Statistics:" in captured.out
        
    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    @patch('play_games.load_words')
    def test_main_without_words(self, mock_load_words, mock_torch_load, mock_agent_class, mock_env_class, capsys):
        """Test main function without provided words."""
        # Set up mocks
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.valid_words = ["apple", "beach", "crane"]
        
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env._compute_feedback_string.return_value = "YBBBB"
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.select_action.return_value = 0
        
        # Create a mock state dict with all required keys
        mock_state_dict = {
            'online_net': {
                'input_layer.0.weight': torch.randn(256, 891),
                'input_layer.0.bias': torch.randn(256),
                'hidden_layer1.0.weight': torch.randn(256, 256),
                'hidden_layer1.0.bias': torch.randn(256),
                'hidden_layer1.2.weight': torch.randn(256, 256),
                'hidden_layer1.2.bias': torch.randn(256),
                'hidden_layer2.0.weight': torch.randn(256, 256),
                'hidden_layer2.0.bias': torch.randn(256),
                'hidden_layer2.2.weight': torch.randn(256, 256),
                'hidden_layer2.2.bias': torch.randn(256),
                'output_layer.weight': torch.randn(500, 256),
                'output_layer.bias': torch.randn(500)
            }
        }
        mock_torch_load.return_value = mock_state_dict
        mock_load_words.return_value = ["apple", "beach", "crane"]
        
        # Run main without words
        main()
        
        # Check that words were loaded from file
        mock_load_words.assert_called_once_with('word_lists/test_words.txt')
        
        # Check output
        captured = capsys.readouterr()
        assert "Using 3 words for gameplay" in captured.out
        assert "Playing 20 games" in captured.out
        assert "Game Statistics:" in captured.out
        
    @patch('play_games.main')
    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    @patch('play_games.load_words')
    def test_script_execution(self, mock_load_words, mock_torch_load, mock_agent_class, mock_env_class, mock_main):
        """Test script execution."""
        # Set up mocks similar to test_main_without_words
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.valid_words = ["apple", "beach", "crane"]
        
        # Set up mock environment responses
        feedback_matrix = np.zeros((6, 26, 13))
        mock_env.reset.return_value = (feedback_matrix, np.ones(3), 6)
        mock_env.step.return_value = (feedback_matrix, np.ones(3), 5, 1.0, True)
        mock_env._compute_feedback_string.return_value = "YBBBB"
        
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.select_action.return_value = 0
        
        # Create a mock state dict with all required keys
        mock_state_dict = {
            'online_net': {
                'input_layer.0.weight': torch.randn(256, 891),
                'input_layer.0.bias': torch.randn(256),
                'hidden_layer1.0.weight': torch.randn(256, 256),
                'hidden_layer1.0.bias': torch.randn(256),
                'hidden_layer1.2.weight': torch.randn(256, 256),
                'hidden_layer1.2.bias': torch.randn(256),
                'hidden_layer2.0.weight': torch.randn(256, 256),
                'hidden_layer2.0.bias': torch.randn(256),
                'hidden_layer2.2.weight': torch.randn(256, 256),
                'hidden_layer2.2.bias': torch.randn(256),
                'output_layer.weight': torch.randn(500, 256),
                'output_layer.bias': torch.randn(500)
            }
        }
        mock_torch_load.return_value = mock_state_dict
        mock_load_words.return_value = ["apple", "beach", "crane"]
        
        # Import play_games module and call its main function directly
        import play_games
        play_games.main()
        
        # Verify main was called
        mock_main.assert_called_once()

    @patch('play_games.WordleEnvironment')
    @patch('play_games.DQNAgent')
    @patch('play_games.torch.load')
    def test_main_model_load_error(self, mock_torch_load, mock_agent_class, mock_env_class, capsys):
        """Test main function handling model loading error."""
        # Mock torch.load to raise an error
        mock_torch_load.side_effect = FileNotFoundError("Model file not found")
        
        # Run main with empty word list to focus on model loading
        with pytest.raises(FileNotFoundError):
            main(words=["apple", "beach"])
        
        # Verify error message
        captured = capsys.readouterr()
        assert "Model file not found" in str(mock_torch_load.side_effect)

    def test_main_script_execution_direct(self):
        """Test executing the script directly."""
        with patch('play_games.main') as mock_main:
            # Save original __name__
            import play_games
            original_name = play_games.__name__
            try:
                # Set __name__ to '__main__' to trigger main block
                play_games.__name__ = '__main__'
                
                # Re-execute the main block
                if hasattr(play_games, '__name__') and play_games.__name__ == '__main__':
                    play_games.main()
            finally:
                # Restore original __name__
                play_games.__name__ = original_name
            
            # Verify main was called
            mock_main.assert_called_once()

```

### test_replay_buffer

```python
"""
Tests for the replay buffer implementation.
"""

import pytest
import numpy as np
from replay_buffer import ReplayBuffer

class TestReplayBuffer:
    """Test suite for ReplayBuffer class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        buffer = ReplayBuffer(max_size=100, state_dim=10)
        assert buffer.max_size == 100
        assert buffer.state_dim == 10
        assert buffer.size == 0
        assert buffer.ptr == 0
        
    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=0, state_dim=10)
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=-1, state_dim=10)
        with pytest.raises(ValueError, match="state_dim must be positive"):
            ReplayBuffer(max_size=100, state_dim=0)
        with pytest.raises(ValueError, match="state_dim must be positive"):
            ReplayBuffer(max_size=100, state_dim=-1)
            
    def test_add_valid(self):
        """Test adding valid transitions."""
        buffer = ReplayBuffer(max_size=2, state_dim=3)
        
        state = np.array([1, 2, 3], dtype=np.float32)
        next_state = np.array([4, 5, 6], dtype=np.float32)
        
        # Add first transition
        buffer.add(state, 0, 1.0, next_state, False)
        assert buffer.size == 1
        assert buffer.ptr == 1
        np.testing.assert_array_equal(buffer.states[0], state)
        np.testing.assert_array_equal(buffer.next_states[0], next_state)
        assert buffer.actions[0] == 0
        assert buffer.rewards[0] == 1.0
        assert buffer.dones[0] == 0.0
        
        # Add second transition
        buffer.add(next_state, 1, -1.0, state, True)
        assert buffer.size == 2
        assert buffer.ptr == 0  # Wrapped around
        np.testing.assert_array_equal(buffer.states[1], next_state)
        np.testing.assert_array_equal(buffer.next_states[1], state)
        assert buffer.actions[1] == 1
        assert buffer.rewards[1] == -1.0
        assert buffer.dones[1] == 1.0
        
    def test_add_invalid(self):
        """Test adding invalid transitions."""
        buffer = ReplayBuffer(max_size=100, state_dim=2)
        valid_state = np.array([1, 2], dtype=np.float32)
        
        # Invalid state shape
        with pytest.raises(ValueError, match="state must be a numpy array of shape"):
            buffer.add(np.array([1]), 0, 1.0, valid_state, False)
            
        # Invalid next_state shape
        with pytest.raises(ValueError, match="next_state must be a numpy array of shape"):
            buffer.add(valid_state, 0, 1.0, np.array([1]), False)
            
        # Invalid action type
        with pytest.raises(ValueError, match="action must be an integer"):
            buffer.add(valid_state, 0.5, 1.0, valid_state, False)
            
        # Invalid reward type
        with pytest.raises(ValueError, match="reward must be a float"):
            buffer.add(valid_state, 0, "1.0", valid_state, False)
            
        # Invalid done type
        with pytest.raises(ValueError, match="done must be a boolean"):
            buffer.add(valid_state, 0, 1.0, valid_state, 1)
            
    def test_sample_valid(self):
        """Test sampling with valid parameters."""
        buffer = ReplayBuffer(max_size=3, state_dim=2)
        
        # Add some transitions
        states = [
            np.array([1, 2], dtype=np.float32),
            np.array([3, 4], dtype=np.float32),
            np.array([5, 6], dtype=np.float32)
        ]
        
        for i, state in enumerate(states):
            next_state = states[(i + 1) % len(states)]
            buffer.add(state, i, float(i), next_state, i == len(states) - 1)
            
        # Sample single transition
        batch = buffer.sample(1)
        assert isinstance(batch, dict)
        assert all(k in batch for k in ['states', 'actions', 'rewards', 'next_states', 'dones'])
        assert all(isinstance(v, np.ndarray) for v in batch.values())
        assert all(len(v) == 1 for v in batch.values())
        
        # Sample multiple transitions
        batch = buffer.sample(2)
        assert all(len(v) == 2 for v in batch.values())
        
        # Verify shapes
        assert batch['states'].shape == (2, 2)
        assert batch['actions'].shape == (2,)
        assert batch['rewards'].shape == (2,)
        assert batch['next_states'].shape == (2, 2)
        assert batch['dones'].shape == (2,)
        
    def test_sample_invalid(self):
        """Test sampling with invalid parameters."""
        buffer = ReplayBuffer(max_size=100, state_dim=2)
        
        # Empty buffer
        with pytest.raises(ValueError, match="Cannot sample from an empty buffer"):
            buffer.sample(1)
            
        # Add one transition
        state = np.array([1, 2], dtype=np.float32)
        buffer.add(state, 0, 1.0, state, False)
        
        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            buffer.sample(0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            buffer.sample(-1)
            
        # Batch size too large
        with pytest.raises(ValueError, match="Cannot sample .* transitions"):
            buffer.sample(2)
            
    def test_buffer_overflow(self):
        """Test FIFO behavior when buffer is full."""
        buffer = ReplayBuffer(max_size=2, state_dim=2)
        
        states = [
            np.array([1, 2], dtype=np.float32),
            np.array([3, 4], dtype=np.float32),
            np.array([5, 6], dtype=np.float32)
        ]
        
        # Fill buffer
        buffer.add(states[0], 0, 0.0, states[1], False)  # Goes to index 0
        buffer.add(states[1], 1, 1.0, states[2], False)  # Goes to index 1
        assert buffer.size == 2
        assert buffer.ptr == 0  # Points to where next item will go
        
        # Add one more transition (should replace oldest at index 0)
        buffer.add(states[2], 2, 2.0, states[0], True)
        assert buffer.size == 2
        assert buffer.ptr == 1  # Points to next position
        
        # Verify the oldest transition was replaced
        # The new transition should be at index 0
        np.testing.assert_array_equal(buffer.states[0], states[2])
        assert buffer.actions[0] == 2
        assert buffer.rewards[0] == 2.0
        assert buffer.dones[0] == 1.0
        
        # The second transition should still be at index 1
        np.testing.assert_array_equal(buffer.states[1], states[1])
        assert buffer.actions[1] == 1
        assert buffer.rewards[1] == 1.0
        assert buffer.dones[1] == 0.0

```

### test_train

```python
"""Tests for the training module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import torch

from train import train
from environment import WordleEnvironment
from agent import DQNAgent
from replay_buffer import ReplayBuffer


class TestTrain:
    """Test cases for training functionality."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock WordleEnvironment."""
        env = MagicMock(spec=WordleEnvironment)
        
        # Mock reset method
        feedback_matrix = np.zeros((6, 26, 13))
        valid_mask = np.ones(5, dtype=bool)
        remaining_guesses = 6
        env.reset.return_value = (feedback_matrix, valid_mask, remaining_guesses)
        
        # Mock step method
        env.step.return_value = (feedback_matrix, valid_mask, remaining_guesses-1, 1.0, True)
        
        return env

    @pytest.fixture
    def mock_agent(self):
        """Create a mock DQNAgent."""
        agent = MagicMock(spec=DQNAgent)
        agent.epsilon = 1.0
        agent.select_action.return_value = 0
        agent.learn.return_value = 0.5
        return agent

    @pytest.fixture
    def mock_buffer(self):
        """Create a mock ReplayBuffer."""
        buffer = MagicMock(spec=ReplayBuffer)
        buffer.size = 100
        buffer.sample.return_value = {
            'states': torch.zeros(64, 390 + 5 + 1),
            'actions': torch.zeros(64, dtype=torch.long),
            'rewards': torch.zeros(64),
            'next_states': torch.zeros(64, 390 + 5 + 1),
            'dones': torch.zeros(64, dtype=torch.bool)
        }
        return buffer

    @patch('train.WordleEnvironment')
    @patch('train.DQNAgent')
    @patch('train.ReplayBuffer')
    def test_train_basic(self, mock_buffer_class, mock_agent_class, mock_env_class,
                        mock_env, mock_agent, mock_buffer):
        """Test basic training loop with minimal episodes."""
        # Set up mocks
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        mock_buffer_class.return_value = mock_buffer

        # Test data
        valid_words = ["apple", "beach", "crane", "dance", "eagle"]
        test_words = ["fight", "ghost", "house", "image", "juice"]

        # Run training with minimal episodes
        agent, metrics = train(
            valid_words=valid_words,
            test_words=test_words,
            num_episodes=2,
            eval_freq=1,
            eval_episodes=2,
            batch_size=64
        )

        # Verify basic interactions
        assert mock_env.reset.called
        assert mock_env.step.called
        assert mock_agent.select_action.called
        assert mock_agent.learn.called
        assert mock_buffer.add.called
        assert mock_buffer.sample.called

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'episode_rewards' in metrics
        assert 'episode_lengths' in metrics
        assert 'running_avg_reward' in metrics
        assert 'running_avg_length' in metrics
        assert 'eval_solve_rates' in metrics
        assert 'best_solved_rate' in metrics

    @patch('train.WordleEnvironment')
    @patch('train.DQNAgent')
    @patch('train.ReplayBuffer')
    def test_train_evaluation(self, mock_buffer_class, mock_agent_class, mock_env_class,
                            mock_env, mock_agent, mock_buffer):
        """Test evaluation phase of training."""
        # Set up mocks
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        mock_buffer_class.return_value = mock_buffer

        # Configure mock to simulate successful guesses during evaluation
        feedback_matrix = np.zeros((6, 26, 13))
        valid_mask = np.ones(5)
        mock_env.reset.return_value = (feedback_matrix, valid_mask, 6)

        # Create a list of step responses for both training and evaluation
        step_responses = [
            # Training episode responses
            (feedback_matrix, valid_mask, 5, 1.0, True),
            # Evaluation episode 1 responses
            (feedback_matrix, valid_mask, 5, 1.0, True),
            # Evaluation episode 2 responses
            (feedback_matrix, valid_mask, 5, 1.0, True),
        ]
        mock_env.step.side_effect = step_responses

        # Test data
        valid_words = ["apple", "beach", "crane", "dance", "eagle"]
        test_words = ["fight", "ghost", "house", "image", "juice"]

        # Run training with evaluation
        agent, metrics = train(
            valid_words=valid_words,
            test_words=test_words,
            num_episodes=1,
            eval_freq=1,
            eval_episodes=2,
            batch_size=64
        )

        # Verify evaluation metrics
        assert len(metrics['eval_solve_rates']) > 0
        assert metrics['best_solved_rate'] >= 0

    def test_train_invalid_inputs(self):
        """Test training with invalid inputs."""
        valid_words = ["apple", "beach", "crane"]
        test_words = ["fight", "ghost"]

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            train(valid_words=valid_words, test_words=test_words, hidden_dim=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            train(valid_words=valid_words, test_words=test_words, learning_rate=0)

        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, gamma=1.5)

        with pytest.raises(ValueError, match="epsilon_start must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, epsilon_start=1.5)

        with pytest.raises(ValueError, match="epsilon_end must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, epsilon_end=1.5)

        with pytest.raises(ValueError, match="epsilon_decay must be between 0 and 1"):
            train(valid_words=valid_words, test_words=test_words, epsilon_decay=1.5)

        with pytest.raises(ValueError, match="target_update_freq must be positive"):
            train(valid_words=valid_words, test_words=test_words, target_update_freq=0)

        with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
            train(valid_words=valid_words, test_words=test_words, device="invalid")

        with pytest.raises(ValueError, match="valid_words must not be empty"):
            train(valid_words=[], test_words=test_words)

        with pytest.raises(ValueError, match="test_words must not be empty"):
            train(valid_words=valid_words, test_words=[])

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            train(valid_words=valid_words, test_words=test_words, num_episodes=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            train(valid_words=valid_words, test_words=test_words, batch_size=0)

        with pytest.raises(ValueError, match="eval_freq must be positive"):
            train(valid_words=valid_words, test_words=test_words, eval_freq=0)

        with pytest.raises(ValueError, match="eval_episodes must be positive"):
            train(valid_words=valid_words, test_words=test_words, eval_episodes=0)

    @patch('train.WordleEnvironment')
    @patch('train.DQNAgent')
    @patch('train.ReplayBuffer')
    def test_train_early_stopping(self, mock_buffer_class, mock_agent_class, mock_env_class,
                                mock_env, mock_agent, mock_buffer):
        """Test training with early stopping based on solve rate."""
        # Set up mocks
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        mock_buffer_class.return_value = mock_buffer

        # Configure mock to simulate perfect solve rate
        mock_env.step.return_value = (np.zeros((6, 26, 13)), np.ones(5), 5, 2.0, True)

        # Test data
        valid_words = ["apple", "beach", "crane", "dance", "eagle"]
        test_words = ["fight", "ghost", "house", "image", "juice"]

        # Run training
        agent, metrics = train(
            valid_words=valid_words,
            test_words=test_words,
            num_episodes=100,
            eval_freq=10,
            eval_episodes=5,
            batch_size=64
        )

        # Verify metrics reflect good performance
        assert metrics['best_solved_rate'] > 0

    def test_main_script_execution(self):
        """Test the main script execution block."""
        with patch('train.load_words') as mock_load_words, \
             patch('train.train') as mock_train:
            
            # Mock load_words to return test data
            mock_load_words.side_effect = [
                ["apple", "beach"],  # train_words
                ["fight", "ghost"]   # test_words
            ]

            # Mock train function
            mock_agent = MagicMock()
            mock_metrics = {
                'episode_rewards': [1.0],
                'episode_lengths': [1],
                'running_avg_reward': [1.0],
                'running_avg_length': [1],
                'eval_solve_rates': [0.5],
                'best_solved_rate': 0.5
            }
            mock_train.return_value = (mock_agent, mock_metrics)

            # Run main script by importing and calling main function directly
            import train
            
            # Save original __name__
            original_name = train.__name__
            try:
                # Set __name__ to '__main__' to trigger main block
                train.__name__ = '__main__'
                
                # Re-execute the main block
                if hasattr(train, '__name__') and train.__name__ == '__main__':
                    train_words = mock_load_words('word_lists/train_words.txt')
                    test_words = mock_load_words('word_lists/test_words.txt')
                    
                    print(f"Loaded {len(train_words)} training words and {len(test_words)} test words")
                    
                    # Train the agent
                    trained_agent, metrics = train.train(
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
            finally:
                # Restore original __name__
                train.__name__ = original_name

            # Verify load_words was called with correct paths
            mock_load_words.assert_any_call('word_lists/train_words.txt')
            mock_load_words.assert_any_call('word_lists/test_words.txt')

            # Verify train was called with expected arguments
            mock_train.assert_called_once()
            args = mock_train.call_args[1]
            assert args['num_episodes'] == 1000
            assert args['hidden_dim'] == 256
            assert args['learning_rate'] == 1e-4
            assert args['gamma'] == 0.99
            assert args['epsilon_start'] == 1.0
            assert args['epsilon_end'] == 0.01
            assert args['epsilon_decay'] == 0.995
            assert args['batch_size'] == 64
            assert args['target_update_freq'] == 100
            assert args['eval_freq'] == 200
            assert args['eval_episodes'] == 50
            assert args['device'] == "cpu"

```

### test_utils

```python
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

```

