# main.py
import argparse
import torch
from train import train
from test import test

def load_words(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def parse_args():
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