import pygame
from game import SnakeGame
from agent import Agent

game = SnakeGame(10)
agent = Agent()
state = game.reset()

print("Training for 200 games...")
while agent.n_games < 200:
    action = agent.get_action(state)
    next_state, reward, done, score = game.step(action)
    agent.remember(state, action, reward, next_state, done)
    agent.train_short_memory(state, action, reward, next_state, done)
    state = next_state
    if done:
        agent.train_long_memory()
        agent.record_game(score)
        if agent.n_games % 20 == 0:
            print(f"Game {agent.n_games} - Score: {score} - Mean Score: {agent.mean_scores[-1]:.2f} - Epsilon: {agent.epsilon:.2f}")
        state = game.reset()
print("Done training test!")
