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