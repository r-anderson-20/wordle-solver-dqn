import random

def load_words(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_words(words, filename):
    with open(filename, 'w') as f:
        for word in words:
            f.write(word + '\n')

def main():
    # Load all words
    all_words = load_words('data/trainlist')
    
    # Shuffle the words
    random.seed(42)  # For reproducibility
    random.shuffle(all_words)
    
    # Split into train (500) and test (500)
    train_words = all_words[:500]
    test_words = all_words[500:1000]
    
    # Save to separate files
    save_words(train_words, 'data/train_words.txt')
    save_words(test_words, 'data/test_words.txt')
    
    print(f"Total words: {len(all_words)}")
    print(f"Training words: {len(train_words)}")
    print(f"Test words: {len(test_words)}")

if __name__ == "__main__":
    main()
