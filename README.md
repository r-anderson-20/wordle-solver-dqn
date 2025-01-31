# Wordle Solver using Deep Q-Learning

This project implements an AI agent that learns to play Wordle using Deep Q-Learning (DQN). The agent achieves a 100% solve rate on test words with an average of 3.0 guesses per word.

## Project Overview

The agent uses reinforcement learning to develop optimal strategies for solving Wordle puzzles. It learns to:
1. Make strategic initial guesses that maximize information gain
2. Use feedback from previous guesses effectively
3. Narrow down possible solutions based on letter positions and frequencies

### Key Features
- Custom Wordle environment implementation
- Deep Q-Network (DQN) with experience replay
- Separate training and testing word sets
- Interactive play mode for human vs. AI comparison
- Pre-trained model included

## Architecture

### Environment (`environment.py`)
- Implements the Wordle game mechanics
- Provides state representation as a 3D matrix (5x26x3):
  - 5 letter positions
  - 26 possible letters
  - 3 feedback types (correct, present, absent)
- Handles word validation and feedback generation

### DQN Agent (`agent.py`)
- Neural network architecture:
  - Input: State vector (feedback matrix + valid word mask + remaining guesses)
  - Hidden layers with ReLU activation
  - Output: Q-values for each possible word
- Epsilon-greedy exploration strategy
- Experience replay for stable learning
- Target network for Q-learning stability

### Training Process (`train.py`)
The training process uses both training and test word sets:
- Training words (`word_lists/train_words.txt`):
  - Used for actual training episodes
  - Agent learns optimal guessing strategies from these words
  - Currently contains 500 words
  - Recommended to keep under 1000 words for efficient training

- Test words (`word_lists/test_words.txt`):
  - Used for periodic evaluation during training
  - Used for final performance testing
  - Currently contains 500 words
  - Recommended to keep under 1000 words for efficient evaluation

The agent is evaluated every 100 episodes during training to track progress and save the best-performing model.

## Results

The trained agent achieves:
- 100% solve rate on test words
- Average of 3.0 guesses per solved puzzle
- Guess distribution:
  - 25% solved in 2 guesses
  - 50% solved in 3 guesses
  - 25% solved in 4 guesses
- Consistent performance across different word patterns

## Installation

1. Clone the repository:
```bash
git clone https://github.com/r-anderson-20/wordle-solver-dqn.git
cd wordle-solver-dqn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train New Model
```bash
python main.py --mode train --num_episodes 1000
```

### Play Games with Trained Model
```bash
python main.py --mode test
```

## Project Structure
```
wordle-solver-DQN/
├── agent.py           # DQN agent implementation
├── environment.py     # Wordle environment
├── main.py           # Main entry point for training and testing
├── train.py          # Training implementation
├── play_games.py     # Interactive gameplay and evaluation
├── utils.py          # Utility functions
├── replay_buffer.py  # Experience replay implementation
├── model/            # Trained model files
│   └── dqn_model_final.pth  # Latest trained model
├── word_lists/       # Word datasets
│   ├── train_words.txt      # Training word set (500 words)
│   └── test_words.txt       # Testing word set (500 words)
└── requirements.txt   # Project dependencies
```

## Dependencies
- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- tqdm >= 4.65.0

## Limitations and Considerations
1. Word List Size:
   - Training and test sets are currently limited to 500 words each
   - Larger word sets are possible but will increase training time
   - Performance may degrade with significantly larger word sets (>1000 words)

2. Training Time:
   - Training for 1000 episodes takes approximately 1-2 minutes
   - Increasing the number of episodes or word list size will increase training time

3. Memory Usage:
   - The experience replay buffer size is proportional to the number of training episodes
   - Large word lists may require more memory for the valid word mask

4. Model Size:
   - The final trained model is approximately 7MB
   - Model size scales with the hidden layer dimensions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
