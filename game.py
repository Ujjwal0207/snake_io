"""
Snake Game Engine
Pure game logic, fully decoupled from rendering.
Provides state encoding for the DQN agent.
"""

import numpy as np
import random

# Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Direction vectors [row_delta, col_delta]
DIR_VECTORS = [
    (-1, 0),   # UP
    (0, 1),    # RIGHT
    (1, 0),    # DOWN
    (0, -1),   # LEFT
]

# Actions: 0=go straight, 1=turn right, 2=turn left
ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2


class SnakeGame:
    """
    Grid-based Snake game with state encoding for RL agents.

    The snake starts at the center of the grid moving upward.
    Apples appear at random unoccupied cells.
    Game ends on wall collision, self-collision, or exceeding max steps.
    """

    def __init__(self, grid_size: int):
        """
        Args:
            grid_size: Side length of the square grid (e.g., 10 means 10x10).
        """
        self.grid_size = grid_size
        self.max_steps = grid_size * grid_size * 2  # prevent infinite loops
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset game to initial state. Snake starts at center, heading UP.

        Returns:
            state: 11-element numpy array encoding the game state.
        """
        mid = self.grid_size // 2
        # Snake stored as list of (row, col) tuples. Index 0 = head.
        self.snake = [
            (mid, mid),        # head
            (mid + 1, mid),    # body
            (mid + 2, mid),    # tail
        ]
        self.direction = UP
        self.score = 0
        self.step_count = 0
        self.done = False
        self.apple = None
        self._place_apple()
        return self.get_state()

    def _place_apple(self):
        """Place apple at a random cell NOT occupied by the snake."""
        occupied = set(self.snake)

        # Collect all free cells
        free_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in occupied:
                    free_cells.append((r, c))

        if not free_cells:
            # Snake fills the entire board — you win!
            self.apple = None
            self.done = True
            return

        self.apple = random.choice(free_cells)

    def step(self, action: int):
        """
        Execute one game step.

        Args:
            action: 0=straight, 1=turn right, 2=turn left (relative to current heading).

        Returns:
            state:  11-element numpy array of the new state.
            reward: float reward signal.
            done:   bool, True if game is over.
            score:  int, current number of apples eaten.
        """
        if self.done:
            return self.get_state(), 0.0, True, self.score

        self.step_count += 1

        # Update direction
        self.direction = self._get_new_direction(action)

        # Calculate new head position
        head_r, head_c = self.snake[0]
        dr, dc = DIR_VECTORS[self.direction]
        new_head = (head_r + dr, head_c + dc)

        # --- Check wall collision ---
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.done = True
            return self.get_state(), -10.0, True, self.score

        # --- Check self collision (exclude tail since it will move) ---
        if new_head in self.snake[:-1]:
            self.done = True
            return self.get_state(), -10.0, True, self.score

        # Move snake: insert new head
        self.snake.insert(0, new_head)

        # --- Check if apple eaten ---
        if self.apple and new_head == self.apple:
            self.score += 1
            reward = 10.0
            self._place_apple()  # New apple; don't remove tail = snake grows
        else:
            self.snake.pop()  # Remove tail (no growth)

            # Small reward for moving toward food, penalty for moving away
            old_dist = abs(head_r - self.apple[0]) + abs(head_c - self.apple[1])
            new_dist = abs(new_head[0] - self.apple[0]) + abs(new_head[1] - self.apple[1])
            reward = 0.1 if new_dist < old_dist else -0.1

        # --- Check max steps exceeded ---
        if self.step_count >= self.max_steps:
            self.done = True
            return self.get_state(), -10.0, True, self.score

        return self.get_state(), reward, False, self.score

    def _get_new_direction(self, action: int) -> int:
        """
        Compute new direction from action relative to current heading.

        Args:
            action: 0=straight, 1=turn right, 2=turn left

        Returns:
            New direction (0-3).
        """
        if action == ACTION_STRAIGHT:
            return self.direction
        elif action == ACTION_RIGHT:
            return (self.direction + 1) % 4
        elif action == ACTION_LEFT:
            return (self.direction + 3) % 4  # +3 ≡ -1 (mod 4)
        return self.direction

    def get_state(self) -> np.ndarray:
        """
        Encode the game state as an 11-feature numpy array.

        Features:
            [0]  danger straight   (wall or body in next cell ahead)
            [1]  danger right      (wall or body to the right)
            [2]  danger left       (wall or body to the left)
            [3]  food is above     (relative to head)
            [4]  food is below
            [5]  food is to left
            [6]  food is to right
            [7]  heading up        (one-hot)
            [8]  heading right
            [9]  heading down
            [10] heading left

        Returns:
            np.ndarray of shape (11,) with float32 values (0.0 or 1.0).
        """
        head_r, head_c = self.snake[0]

        # Relative direction vectors
        straight = DIR_VECTORS[self.direction]
        right_dir = DIR_VECTORS[(self.direction + 1) % 4]
        left_dir = DIR_VECTORS[(self.direction + 3) % 4]

        # Danger checks
        danger_straight = self._is_danger(head_r + straight[0], head_c + straight[1])
        danger_right = self._is_danger(head_r + right_dir[0], head_c + right_dir[1])
        danger_left = self._is_danger(head_r + left_dir[0], head_c + left_dir[1])

        # Food direction (absolute)
        if self.apple:
            food_up = 1.0 if self.apple[0] < head_r else 0.0
            food_down = 1.0 if self.apple[0] > head_r else 0.0
            food_left = 1.0 if self.apple[1] < head_c else 0.0
            food_right = 1.0 if self.apple[1] > head_c else 0.0
        else:
            food_up = food_down = food_left = food_right = 0.0

        # Direction one-hot
        dir_up = 1.0 if self.direction == UP else 0.0
        dir_right = 1.0 if self.direction == RIGHT else 0.0
        dir_down = 1.0 if self.direction == DOWN else 0.0
        dir_left = 1.0 if self.direction == LEFT else 0.0

        return np.array([
            danger_straight, danger_right, danger_left,
            food_up, food_down, food_left, food_right,
            dir_up, dir_right, dir_down, dir_left,
        ], dtype=np.float32)

    def _is_danger(self, r: int, c: int) -> float:
        """Check if a cell is dangerous (wall or snake body)."""
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return 1.0  # wall
        if (r, c) in self.snake:
            return 1.0  # body
        return 0.0

    def get_snapshot(self) -> dict:
        """
        Get a rendering-friendly snapshot of the current game state.

        Returns:
            dict with keys: snake, apple, grid_size, score, done, direction
        """
        return {
            "snake": list(self.snake),
            "apple": self.apple,
            "grid_size": self.grid_size,
            "score": self.score,
            "done": self.done,
            "direction": self.direction,
        }
