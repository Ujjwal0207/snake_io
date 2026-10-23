import pygame
import time
from game import SnakeGame
from renderer import GameRenderer

pygame.init()
pygame.font.init()
game = SnakeGame(10)
renderer = GameRenderer(10)

# Force the snake and apple to specific positions
game.snake = [(5, 5), (6, 5), (7, 5)]
game.direction = 0 # UP
game.apple = (4, 5) # Directly above the head
game.score = 0

snapshot = game.get_snapshot()
agent_stats = {"n_games": 1, "score": 0, "high_score": 0, "epsilon": 0, "mean_scores": [0], "scores": [0], "loss": 0, "snake_length": 3}
renderer.render(snapshot, agent_stats)
pygame.display.flip()
time.sleep(1)

# Step to eat the apple
game.step(0) # straight

snapshot = game.get_snapshot()
agent_stats["snake_length"] = len(game.snake)
agent_stats["score"] = game.score
renderer.render(snapshot, agent_stats)
pygame.display.flip()
time.sleep(1)

pygame.quit()
