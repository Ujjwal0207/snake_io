import pygame
import os
from game import SnakeGame
from agent import Agent
from model import save_model

pygame.init()
pygame.font.init()
game = SnakeGame(10)
agent = Agent()
state = game.reset()

print("Pre-training AI (250 games) so it gets smart instantly...")
while agent.n_games < 250:
    action = agent.get_action(state)
    next_state, reward, done, score = game.step(action)
    agent.remember(state, action, reward, next_state, done)
    agent.train_short_memory(state, action, reward, next_state, done)
    state = next_state
    if done:
        agent.train_long_memory()
        agent.record_game(score)
        if agent.n_games % 50 == 0:
            print(f"Game {agent.n_games} - Epsilon: {agent.epsilon:.2f} - High Score: {agent.high_score}")
        state = game.reset()

print(f"Pre-training complete! High score achieved: {agent.high_score}")
os.makedirs('model', exist_ok=True)
agent.save("model/snake_dqn.pth")
print("Saved smart model to model/snake_dqn.pth")
