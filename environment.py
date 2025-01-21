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