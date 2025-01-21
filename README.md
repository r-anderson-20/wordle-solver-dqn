# Wordle Solver using Deep Q-Learning

This project implements an AI agent that learns to play Wordle using Deep Q-Learning (DQN). The agent achieves a 100% solve rate on test words with an average of 3.42 guesses per word.

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
- Separate training (500 words) and testing (500 words) sets
- Periodic evaluation on test set
- Performance metrics tracking:
  - Solve rate
  - Average number of guesses
  - Training rewards

## Results

The trained agent achieves:
- 100% solve rate on test words
- Average of 3.42 guesses per solved puzzle
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

### Play with Pre-trained Model
```bash
python play_games.py
```

### Train New Model
```bash
python train.py
```

### Test Model Performance
```bash
python test.py
```

## Project Structure
```
wordle-solver-DQN/
├── agent.py           # DQN agent implementation
├── environment.py     # Wordle environment
├── train.py          # Training script
├── play_games.py     # Interactive gameplay
├── test.py           # Testing utilities
├── replay_buffer.py  # Experience replay implementation
├── data/
│   ├── train_words.txt  # Training word set
│   └── test_words.txt   # Testing word set
└── requirements.txt   # Project dependencies
```

## Dependencies
- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- tqdm >= 4.65.0

## Training Details

The agent is trained using:
- Experience replay buffer (size: 10,000)
- Epsilon-greedy exploration (start: 1.0, end: 0.01, decay: 0.995)
- Learning rate: 1e-4
- Discount factor (gamma): 0.99
- Hidden layer size: 256 neurons
- Target network update frequency: 100 episodes
- Batch size: 64

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by the original Wordle game by Josh Wardle
- Built using PyTorch for deep learning implementation
