"""
Snake AI — Main Entry Point
Asks for grid size, initializes the game + agent + renderer,
and runs the training loop with live visualization.

Usage:
    python main.py
"""

import pygame
import sys
import time

from game import SnakeGame
from agent import Agent
from renderer import GameRenderer


def get_grid_size() -> int:
    """
    Interactive console prompt to ask the user for grid size.
    Validates input and returns an integer between 5 and 30.
    """
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║     🐍 Snake AI — Deep Q-Network Trainer     ║")
    print("║                                              ║")
    print("║  The AI will learn to play Snake by itself    ║")
    print("║  using reinforcement learning (DQN).         ║")
    print("║                                              ║")
    print("║  Watch it go from random moves to strategic  ║")
    print("║  apple hunting over hundreds of games!       ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    while True:
        try:
            size = int(input("Enter the grid size (5 to 30, recommended 10): "))
            if 5 <= size <= 30:
                return size
            else:
                print("⚠️  Please enter a number between 5 and 30.")
        except ValueError:
            print("⚠️  Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)


def main():
    """Main training loop with Pygame visualization."""

    # --- Ask for grid size ---
    grid_size = get_grid_size()
    print(f"\n✅ Starting training on a {grid_size}×{grid_size} grid...")
    print("   Close the window or press Q/ESC to stop.\n")

    # --- Initialize Pygame ---
    pygame.init()
    pygame.font.init()

    # --- Initialize components ---
    game = SnakeGame(grid_size)
    agent = Agent(lr=0.001, gamma=0.9)

    # Attempt to load a pre-trained model
    if agent.load("model/snake_dqn.pth"):
        print("✅ Loaded pre-trained model! The AI is already smart.")
        agent.n_games = 150  # Fast-forward so epsilon is low (0.01)

    renderer = GameRenderer(grid_size)

    # --- Training settings ---
    clock = pygame.time.Clock()
    speed = 10           # FPS (adjustable with UP/DOWN keys) — slow so you can see growth
    max_speed = False    # If True, skip rendering for max training speed
    running = True
    paused = False

    # --- Training loop ---
    state = game.reset()

    print("🎮 Controls:")
    print("   UP/DOWN  — Adjust visualization speed")
    print("   SPACE    — Pause/Resume")
    print("   M        — Toggle max speed (no rendering)")
    print("   R        — Reset agent (clear learned weights)")
    print("   Q/ESC    — Quit")
    print()

    while running:
        # --- Handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"⏸️  {status}")
                elif event.key == pygame.K_UP:
                    speed = min(500, speed + 10)
                    print(f"⚡ Speed: {speed} FPS")
                elif event.key == pygame.K_DOWN:
                    speed = max(5, speed - 10)
                    print(f"🐢 Speed: {speed} FPS")
                elif event.key == pygame.K_m:
                    max_speed = not max_speed
                    mode = "MAX SPEED (no rendering)" if max_speed else "VISUAL MODE"
                    print(f"🔄 Mode: {mode}")
                elif event.key == pygame.K_r:
                    agent = Agent(lr=0.001, gamma=0.9)
                    game.reset()
                    print("🔄 Agent reset — starting fresh!")

        if paused:
            clock.tick(10)
            continue

        # --- AI decides action ---
        action = agent.get_action(state)

        # --- Execute action ---
        next_state, reward, done, score = game.step(action)

        # --- Train on this step (short memory) ---
        agent.train_short_memory(state, action, reward, next_state, done)

        # --- Remember the experience ---
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        # --- Episode ended ---
        if done:
            # --- Show GAME OVER frame before resetting ---
            if not max_speed:
                snapshot = game.get_snapshot()
                agent_stats = {
                    "n_games": agent.n_games,
                    "score": game.score,
                    "high_score": agent.high_score,
                    "epsilon": agent.epsilon,
                    "mean_scores": agent.mean_scores,
                    "scores": agent.scores,
                    "loss": agent.last_loss,
                    "snake_length": len(game.snake),
                }
                renderer.render(snapshot, agent_stats)
                pygame.display.flip()
                pygame.time.wait(300)  # Brief pause so you can see GAME OVER

            # Train on replay buffer (long memory)
            agent.train_long_memory()
            agent.record_game(score)

            # Print progress
            n = agent.n_games
            if n <= 20 or n % 10 == 0:
                print(
                    f"  Game {n:>4d}  │  "
                    f"Score: {score:>3d}  │  "
                    f"Length: {len(game.snake):>3d}  │  "
                    f"High: {agent.high_score:>3d}  │  "
                    f"Avg: {agent.mean_scores[-1]:>6.1f}  │  "
                    f"ε: {agent.epsilon:.3f}  │  "
                    f"Loss: {agent.last_loss:.4f}"
                )

            # Reset game for next episode
            state = game.reset()

        # --- Render ---
        if not max_speed:
            snapshot = game.get_snapshot()
            agent_stats = {
                "n_games": agent.n_games,
                "score": game.score,
                "high_score": agent.high_score,
                "epsilon": agent.epsilon,
                "mean_scores": agent.mean_scores,
                "scores": agent.scores,
                "loss": agent.last_loss,
                "snake_length": len(game.snake),
            }
            renderer.render(snapshot, agent_stats)
            clock.tick(speed)
        else:
            # In max speed mode, still render occasionally to keep window responsive
            if agent.n_games % 50 == 0 and done:
                snapshot = game.get_snapshot()
                agent_stats = {
                    "n_games": agent.n_games,
                    "score": game.score,
                    "high_score": agent.high_score,
                    "epsilon": agent.epsilon,
                    "mean_scores": agent.mean_scores,
                    "scores": agent.scores,
                    "loss": agent.last_loss,
                    "snake_length": len(game.snake),
                }
                renderer.render(snapshot, agent_stats)

    # --- Cleanup ---
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Total games:  {agent.n_games}")
    print(f"  High score:   {agent.high_score}")
    if agent.mean_scores:
        print(f"  Final avg:    {agent.mean_scores[-1]:.1f}")
    print(f"{'='*50}")

    agent.save()
    print("💾 Model saved to model/snake_dqn.pth")

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
