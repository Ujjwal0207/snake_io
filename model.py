"""
Deep Q-Network (DQN) Model and Trainer
Neural network architecture and training logic using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os


class DQN(nn.Module):
    """
    Deep Q-Network with 3 fully connected layers.
    Input: 11-feature state vector
    Output: Q-values for 3 actions (straight, right, left)
    """

    def __init__(self, input_size: int = 11, hidden1: int = 256, hidden2: int = 128, output_size: int = 3):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNTrainer:
    """
    Handles training the DQN using the Bellman equation.
    Uses MSE loss between predicted Q-values and target Q-values.
    """

    def __init__(self, model: DQN, lr: float = 0.001, gamma: float = 0.9):
        """
        Args:
            model:  The DQN model to train.
            lr:     Learning rate for Adam optimizer.
            gamma:  Discount factor for future rewards.
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Perform a single training step on a batch of experiences.

        Args:
            states:      torch.Tensor of shape (batch, 11)
            actions:     torch.Tensor of shape (batch,) — action indices
            rewards:     torch.Tensor of shape (batch,)
            next_states: torch.Tensor of shape (batch, 11)
            dones:       torch.Tensor of shape (batch,) — 1 if terminal, 0 otherwise
        """
        # Predict Q-values for current states
        pred_q = self.model(states)  # (batch, 3)

        # Clone predictions as our target
        target_q = pred_q.clone().detach()

        # Compute target using Bellman equation
        with torch.no_grad():
            next_q = self.model(next_states)  # (batch, 3)
            max_next_q = torch.max(next_q, dim=1)[0]  # (batch,)

        for i in range(len(states)):
            q_new = rewards[i]
            if not dones[i]:
                q_new = rewards[i] + self.gamma * max_next_q[i]
            target_q[i][actions[i].long()] = q_new

        # Backprop
        self.optimizer.zero_grad()
        loss = self.criterion(pred_q, target_q)
        loss.backward()
        self.optimizer.step()

        return loss.item()


def save_model(model: DQN, filepath: str = "model/snake_dqn.pth"):
    """Save the model weights to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model(model: DQN, filepath: str = "model/snake_dqn.pth") -> bool:
    """
    Load model weights from disk.
    Returns True if loaded successfully, False if file not found.
    """
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, weights_only=True))
        return True
    return False
