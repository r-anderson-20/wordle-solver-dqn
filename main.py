"""
Main entry point for training and testing the Wordle DQN agent.
Provides command-line interface for running different modes of operation
and managing model files.
"""

import argparse
import torch
from train import train
from test import test

def load_words(filename):
    """
    Load words from a text file.
    
    Args:
        filename (str): Path to the text file containing words
        
    Returns:
        list[str]: List of words from the file
    """
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

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
    parser.add_argument("--model_path", type=str, default="dqn_model.pth")
    parser.add_argument("--train_words", type=str, default="data/train_words.txt")
    parser.add_argument("--test_words", type=str, default="data/test_words.txt")
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
        print(f"Loaded {len(train_words)} training words")
        
        trained_agent = train(
            valid_words=train_words,
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
        # For testing, we need both train and test words
        train_words = load_words(args.train_words)  # For valid guesses
        test_words = load_words(args.test_words)    # For secret words
        print(f"Loaded {len(train_words)} training words and {len(test_words)} test words")
        
        success_rate, avg_guesses = test(
            train_words=train_words,  # Words the agent can use as guesses
            test_words=test_words,    # Words to test on
            model_path=args.model_path,
            hidden_dim=args.hidden_dim,
            device=args.device
        )
        
        print(f"\nTest Results:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Guesses (when solved): {avg_guesses:.2f}")

if __name__ == "__main__":
    main()