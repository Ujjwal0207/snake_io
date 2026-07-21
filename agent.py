"""
DQN Reinforcement Learning Agent
Handles experience replay, epsilon-greedy action selection,
and orchestrates training of the neural network.
"""

import torch
import numpy as np
import random
from collections import deque

from model import DQN, DQNTrainer, save_model, load_model

# Maximum number of experiences stored in replay buffer
MAX_MEMORY = 100_000
BATCH_SIZE = 256


class Agent:
    """
    Deep Q-Learning agent that learns to play Snake.

    The agent uses:
    - Epsilon-greedy policy for exploration vs exploitation
    - Experience replay buffer for stable training
    - A DQN neural network to predict Q-values
    """

    def __init__(self, lr: float = 0.001, gamma: float = 0.9):
        """
        Args:
            lr:    Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
        """
        self.n_games = 0           # Total games played
        self.epsilon = 0           # Will be computed dynamically
        self.gamma = gamma
        self.lr = lr

        # Experience replay buffer
        self.memory = deque(maxlen=MAX_MEMORY)

        # Neural network and trainer
        self.model = DQN(input_size=11, hidden1=256, hidden2=128, output_size=3)
        self.trainer = DQNTrainer(self.model, lr=self.lr, gamma=self.gamma)

        # Track training stats
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.high_score = 0
        self.last_loss = 0.0

    @property
    def current_epsilon(self) -> float:
        """
        Compute epsilon for epsilon-greedy exploration.
        Starts at 1.0 (100% random) and decays to 0.01 over 150 games.
        """
        return max(0.01, 1.0 - self.n_games / 150.0)

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay buffer.

        Args:
            state:      np.ndarray of shape (11,)
            action:     int (0, 1, or 2)
            reward:     float
            next_state: np.ndarray of shape (11,)
            done:       bool
        """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current game state as numpy array of shape (11,).

        Returns:
            Action index: 0=straight, 1=right, 2=left
        """
        self.epsilon = self.current_epsilon

        # Exploration: random action
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        # Exploitation: use neural network
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state_tensor)
        return torch.argmax(prediction).item()

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train on a single experience (online learning after each step).
        """
        self._train_batch(
            np.array([state]),
            np.array([action]),
            np.array([reward]),
            np.array([next_state]),
            np.array([done]),
        )

    def train_long_memory(self):
        """
        Train on a random batch from the experience replay buffer.
        Called at the end of each game episode.
        """
        if len(self.memory) < BATCH_SIZE:
            mini_batch = list(self.memory)
        else:
            mini_batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self._train_batch(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def _train_batch(self, states, actions, rewards, next_states, dones):
        """Convert numpy arrays to tensors and run one training step."""
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        self.last_loss = self.trainer.train_step(
            states_t, actions_t, rewards_t, next_states_t, dones_t
        )

    def record_game(self, score: int):
        """
        Record the result of a completed game.

        Args:
            score: Number of apples eaten in the game.
        """
        self.n_games += 1
        self.scores.append(score)
        self.total_score += score

        if score > self.high_score:
            self.high_score = score
            save_model(self.model)

        # Running mean over last 100 games
        window = self.scores[-100:]
        mean = sum(window) / len(window)
        self.mean_scores.append(mean)

    def save(self, filepath: str = "model/snake_dqn.pth"):
        """Save model weights to disk."""
        save_model(self.model, filepath)

    def load(self, filepath: str = "model/snake_dqn.pth") -> bool:
        """Load model weights from disk. Returns True if successful."""
        return load_model(self.model, filepath)
