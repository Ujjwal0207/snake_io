"""
Pygame Renderer
Handles all visual rendering: grid, snake, apple, stats overlay, and score chart.
Premium dark theme with neon glow effects.
"""

import pygame
import math

# --- Color Palette ---
BG_PRIMARY = (10, 14, 23)
BG_GRID = (17, 24, 39)
GRID_LINE = (30, 41, 59)
SNAKE_HEAD = (0, 255, 136)
SNAKE_BODY = (0, 204, 106)
SNAKE_BODY_ALT = (0, 180, 90)
SNAKE_TAIL = (0, 140, 70)
APPLE_COLOR = (239, 68, 68)
APPLE_GLOW = (239, 68, 68, 80)
TEXT_PRIMARY = (241, 245, 249)
TEXT_SECONDARY = (148, 163, 184)
TEXT_MUTED = (100, 116, 139)
NEON_GREEN = (0, 255, 136)
NEON_PURPLE = (168, 85, 247)
NEON_BLUE = (59, 130, 246)
NEON_ORANGE = (249, 115, 22)
NEON_RED = (239, 68, 68)
PANEL_BG = (17, 24, 39, 200)
BORDER_COLOR = (30, 41, 59)


class GameRenderer:
    """
    Renders the Snake game and training statistics using Pygame.
    """

    # Layout constants
    STATS_PANEL_WIDTH = 300
    HEADER_HEIGHT = 60
    PADDING = 20
    MIN_CELL_SIZE = 12
    MAX_CELL_SIZE = 40

    def __init__(self, grid_size: int):
        """
        Initialize the renderer for the given grid size.

        Args:
            grid_size: Number of cells per side of the game grid.
        """
        self.grid_size = grid_size

        # Calculate cell size to fit nicely on screen
        screen_info = pygame.display.Info()
        available_height = screen_info.current_h - 150
        available_width = screen_info.current_w - self.STATS_PANEL_WIDTH - 100

        cell_from_height = (available_height - self.HEADER_HEIGHT - self.PADDING * 3) // grid_size
        cell_from_width = (available_width - self.PADDING * 3) // grid_size
        self.cell_size = max(self.MIN_CELL_SIZE, min(self.MAX_CELL_SIZE, cell_from_height, cell_from_width))

        # Calculate dimensions
        self.game_area_size = self.cell_size * grid_size
        self.window_width = self.game_area_size + self.STATS_PANEL_WIDTH + self.PADDING * 3
        self.window_height = self.game_area_size + self.HEADER_HEIGHT + self.PADDING * 2

        # Ensure minimum size
        self.window_width = max(self.window_width, 800)
        self.window_height = max(self.window_height, 600)

        # Game area offset (centered in available space)
        self.game_x = self.PADDING
        self.game_y = self.HEADER_HEIGHT + self.PADDING

        # Stats panel position
        self.stats_x = self.game_x + self.game_area_size + self.PADDING

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("🐍 Snake AI — Deep Q-Network Training")

        # Fonts
        self._init_fonts()

        # Animation state
        self.apple_pulse = 0
        self.frame_count = 0

        # Death overlay font
        try:
            self.font_death = pygame.font.SysFont("Inter", 32, bold=True)
        except Exception:
            self.font_death = pygame.font.Font(None, 40)

    def _init_fonts(self):
        """Initialize fonts with fallbacks."""
        try:
            self.font_title = pygame.font.SysFont("Inter", 22, bold=True)
            self.font_stat_label = pygame.font.SysFont("Inter", 12)
            self.font_stat_value = pygame.font.SysFont("JetBrains Mono", 28, bold=True)
            self.font_stat_value_sm = pygame.font.SysFont("JetBrains Mono", 18, bold=True)
            self.font_small = pygame.font.SysFont("Inter", 11)
            self.font_header = pygame.font.SysFont("Inter", 16, bold=True)
        except Exception:
            # Fallback to default font
            self.font_title = pygame.font.Font(None, 28)
            self.font_stat_label = pygame.font.Font(None, 16)
            self.font_stat_value = pygame.font.Font(None, 36)
            self.font_stat_value_sm = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 14)
            self.font_header = pygame.font.Font(None, 20)

    def render(self, snapshot: dict, agent_stats: dict):
        """
        Render one frame of the game + stats.

        Args:
            snapshot:    dict from SnakeGame.get_snapshot()
            agent_stats: dict with keys: n_games, score, high_score, epsilon,
                         mean_scores, scores, loss, snake_length
        """
        self.frame_count += 1
        self.apple_pulse = (math.sin(self.frame_count * 0.1) + 1) / 2  # 0 to 1

        # Clear screen
        self.screen.fill(BG_PRIMARY)

        # Draw components
        self._draw_header(agent_stats)
        self._draw_game_area(snapshot)
        self._draw_stats_panel(agent_stats)
        self._draw_score_chart(agent_stats)

        # Draw death overlay if game is over
        if snapshot.get("done", False):
            self._draw_death_overlay(snapshot)

        pygame.display.flip()

    def _draw_header(self, stats: dict):
        """Draw the top header bar."""
        # Header background
        header_rect = pygame.Rect(0, 0, self.window_width, self.HEADER_HEIGHT)
        pygame.draw.rect(self.screen, BG_GRID, header_rect)
        pygame.draw.line(self.screen, BORDER_COLOR, (0, self.HEADER_HEIGHT), (self.window_width, self.HEADER_HEIGHT))

        # Title
        title = self.font_header.render("🐍 Snake AI — DQN Training", True, TEXT_PRIMARY)
        self.screen.blit(title, (self.PADDING, 12))

        # Status indicator
        is_training = True
        status_color = NEON_GREEN if is_training else NEON_ORANGE
        status_text = "TRAINING" if is_training else "PAUSED"

        # Pulsing dot
        dot_alpha = int(128 + 127 * math.sin(self.frame_count * 0.08))
        dot_x = self.PADDING
        dot_y = 40
        pygame.draw.circle(self.screen, status_color, (dot_x + 4, dot_y + 4), 4)

        status_surf = self.font_small.render(f"  {status_text}  •  Grid: {self.grid_size}×{self.grid_size}", True, TEXT_SECONDARY)
        self.screen.blit(status_surf, (dot_x + 12, dot_y))

        # Network info (right side)
        net_text = self.font_small.render("DQN: 11 → 256 → 128 → 3", True, TEXT_MUTED)
        self.screen.blit(net_text, (self.window_width - net_text.get_width() - self.PADDING, 38))

    def _draw_game_area(self, snapshot: dict):
        """Draw the game grid, snake, and apple."""
        # Game area background
        game_rect = pygame.Rect(self.game_x, self.game_y, self.game_area_size, self.game_area_size)
        pygame.draw.rect(self.screen, BG_GRID, game_rect, border_radius=8)

        # Grid lines
        for i in range(1, self.grid_size):
            # Vertical
            x = self.game_x + i * self.cell_size
            pygame.draw.line(self.screen, GRID_LINE, (x, self.game_y), (x, self.game_y + self.game_area_size))
            # Horizontal
            y = self.game_y + i * self.cell_size
            pygame.draw.line(self.screen, GRID_LINE, (self.game_x, y), (self.game_x + self.game_area_size, y))

        # Border
        pygame.draw.rect(self.screen, BORDER_COLOR, game_rect, 2, border_radius=8)

        # Draw apple
        if snapshot["apple"]:
            self._draw_apple(snapshot["apple"])

        # Draw snake
        self._draw_snake(snapshot["snake"])

    def _draw_apple(self, apple):
        """Draw apple with pulsing glow effect."""
        r, c = apple
        cx = self.game_x + c * self.cell_size + self.cell_size // 2
        cy = self.game_y + r * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 2

        # Glow
        glow_radius = int(radius + 4 + 3 * self.apple_pulse)
        glow_surface = pygame.Surface((glow_radius * 4, glow_radius * 4), pygame.SRCALPHA)
        glow_color = (239, 68, 68, int(40 + 30 * self.apple_pulse))
        pygame.draw.circle(glow_surface, glow_color, (glow_radius * 2, glow_radius * 2), glow_radius)
        self.screen.blit(glow_surface, (cx - glow_radius * 2, cy - glow_radius * 2))

        # Apple body
        pygame.draw.circle(self.screen, APPLE_COLOR, (cx, cy), radius)

        # Highlight
        highlight_radius = max(2, radius // 3)
        pygame.draw.circle(self.screen, (255, 120, 120), (cx - radius // 4, cy - radius // 4), highlight_radius)

    def _draw_snake(self, snake):
        """Draw the snake with gradient coloring from head to tail."""
        n = len(snake)
        for i, (r, c) in enumerate(snake):
            x = self.game_x + c * self.cell_size
            y = self.game_y + r * self.cell_size
            pad = 1

            if i == 0:
                # Head — brightest with glow
                color = SNAKE_HEAD
                glow_surface = pygame.Surface((self.cell_size + 8, self.cell_size + 8), pygame.SRCALPHA)
                pygame.draw.rect(
                    glow_surface, (0, 255, 136, 40),
                    (0, 0, self.cell_size + 8, self.cell_size + 8),
                    border_radius=4
                )
                self.screen.blit(glow_surface, (x - 4, y - 4))
            else:
                # Body — gradient from bright to dim
                t = i / max(1, n - 1)
                r_c = int(SNAKE_BODY[0] * (1 - t) + SNAKE_TAIL[0] * t)
                g_c = int(SNAKE_BODY[1] * (1 - t) + SNAKE_TAIL[1] * t)
                b_c = int(SNAKE_BODY[2] * (1 - t) + SNAKE_TAIL[2] * t)
                color = (r_c, g_c, b_c)

            rect = pygame.Rect(x + pad, y + pad, self.cell_size - pad * 2, self.cell_size - pad * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _draw_death_overlay(self, snapshot: dict):
        """Draw a red GAME OVER overlay on the game area when the snake dies."""
        # Semi-transparent red overlay
        overlay = pygame.Surface((self.game_area_size, self.game_area_size), pygame.SRCALPHA)
        overlay.fill((239, 68, 68, 60))
        self.screen.blit(overlay, (self.game_x, self.game_y))

        # GAME OVER text
        text = self.font_death.render("GAME OVER", True, NEON_RED)
        text_x = self.game_x + (self.game_area_size - text.get_width()) // 2
        text_y = self.game_y + (self.game_area_size - text.get_height()) // 2 - 15
        self.screen.blit(text, (text_x, text_y))

        # Score + Length info
        info = self.font_stat_value_sm.render(
            f"Score: {snapshot['score']}  •  Length: {len(snapshot['snake'])}",
            True, TEXT_PRIMARY
        )
        info_x = self.game_x + (self.game_area_size - info.get_width()) // 2
        info_y = text_y + text.get_height() + 8
        self.screen.blit(info, (info_x, info_y))

    def _draw_stats_panel(self, stats: dict):
        """Draw the stats sidebar with all training metrics."""
        x = self.stats_x
        y = self.game_y
        w = self.STATS_PANEL_WIDTH - self.PADDING

        # --- Episode ---
        y = self._draw_stat_card(x, y, w, "EPISODE", str(stats.get("n_games", 0)), NEON_BLUE)
        y += 8

        # --- Score & High Score (side by side) ---
        half_w = (w - 8) // 2
        self._draw_stat_card(x, y, half_w, "SCORE", str(stats.get("score", 0)), NEON_GREEN)
        self._draw_stat_card(x + half_w + 8, y, half_w, "HIGH SCORE", str(stats.get("high_score", 0)), NEON_ORANGE)
        y += 90 + 8

        # --- Snake Length ---
        snake_len = stats.get("snake_length", 3)
        y = self._draw_stat_card(x, y, w, "SNAKE LENGTH", str(snake_len), NEON_GREEN)
        y += 8

        # --- Epsilon ---
        epsilon_val = stats.get("epsilon", 0)
        y = self._draw_stat_card(x, y, w, "EXPLORATION (ε)", f"{epsilon_val:.3f}", NEON_PURPLE)
        y += 8

        # --- Mean Score ---
        mean_scores = stats.get("mean_scores", [])
        mean_val = f"{mean_scores[-1]:.1f}" if mean_scores else "0.0"
        y = self._draw_stat_card(x, y, w, "AVG SCORE (LAST 100)", mean_val, NEON_GREEN)
        y += 8

        # --- Loss ---
        loss_val = stats.get("loss", 0)
        y = self._draw_stat_card(x, y, w, "LOSS", f"{loss_val:.4f}", NEON_RED)
        y += 8

        # --- Network Architecture ---
        self._draw_network_card(x, y, w)

    def _draw_stat_card(self, x, y, w, label, value, color, h=82):
        """Draw a single stat card. Returns the bottom y coordinate."""
        # Card background
        card_rect = pygame.Rect(x, y, w, h)
        card_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card_surface, (17, 24, 39, 180), (0, 0, w, h), border_radius=10)
        pygame.draw.rect(card_surface, (*BORDER_COLOR, 120), (0, 0, w, h), 1, border_radius=10)
        self.screen.blit(card_surface, (x, y))

        # Label
        label_surf = self.font_stat_label.render(label, True, TEXT_MUTED)
        self.screen.blit(label_surf, (x + 14, y + 12))

        # Value
        value_surf = self.font_stat_value_sm.render(value, True, color)
        self.screen.blit(value_surf, (x + 14, y + 34))

        return y + h

    def _draw_network_card(self, x, y, w):
        """Draw neural network architecture visualization."""
        h = 60
        card_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card_surface, (17, 24, 39, 180), (0, 0, w, h), border_radius=10)
        pygame.draw.rect(card_surface, (*BORDER_COLOR, 120), (0, 0, w, h), 1, border_radius=10)
        self.screen.blit(card_surface, (x, y))

        label_surf = self.font_stat_label.render("NEURAL NETWORK", True, TEXT_MUTED)
        self.screen.blit(label_surf, (x + 14, y + 10))

        # Layer visualization
        layers = [("11", NEON_BLUE), ("256", NEON_PURPLE), ("128", NEON_PURPLE), ("3", NEON_GREEN)]
        total_w = len(layers) * 40 + (len(layers) - 1) * 20
        start_x = x + (w - total_w) // 2
        cy = y + 40

        for i, (size, color) in enumerate(layers):
            cx = start_x + i * 60 + 20
            pygame.draw.circle(self.screen, color, (cx, cy), 12)
            size_surf = self.font_small.render(size, True, BG_PRIMARY)
            self.screen.blit(size_surf, (cx - size_surf.get_width() // 2, cy - size_surf.get_height() // 2))

            # Arrow
            if i < len(layers) - 1:
                arrow_x = cx + 16
                pygame.draw.line(self.screen, TEXT_MUTED, (arrow_x, cy), (arrow_x + 20, cy), 1)
                # Arrowhead
                pygame.draw.polygon(self.screen, TEXT_MUTED, [
                    (arrow_x + 20, cy),
                    (arrow_x + 16, cy - 3),
                    (arrow_x + 16, cy + 3),
                ])

    def _draw_score_chart(self, stats: dict):
        """Draw a mini line chart of scores over time at the bottom of the stats panel."""
        scores = stats.get("scores", [])
        mean_scores = stats.get("mean_scores", [])

        if len(scores) < 2:
            return

        x = self.stats_x
        y = self.game_y + self.game_area_size - 130
        w = self.STATS_PANEL_WIDTH - self.PADDING
        h = 120

        # Card background
        card_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card_surface, (17, 24, 39, 180), (0, 0, w, h), border_radius=10)
        pygame.draw.rect(card_surface, (*BORDER_COLOR, 120), (0, 0, w, h), 1, border_radius=10)
        self.screen.blit(card_surface, (x, y))

        label_surf = self.font_stat_label.render("SCORE HISTORY", True, TEXT_MUTED)
        self.screen.blit(label_surf, (x + 14, y + 10))

        # Chart area
        chart_x = x + 14
        chart_y = y + 30
        chart_w = w - 28
        chart_h = h - 44

        # Get last N scores that fit
        max_points = min(len(scores), chart_w)
        display_scores = scores[-max_points:]
        display_means = mean_scores[-max_points:] if mean_scores else []

        if not display_scores:
            return

        max_score = max(max(display_scores), 1)

        # Draw score dots (individual games)
        if len(display_scores) > 1:
            step = chart_w / max(1, len(display_scores) - 1)
            points = []
            for i, s in enumerate(display_scores):
                px = chart_x + i * step
                py = chart_y + chart_h - (s / max_score) * chart_h
                points.append((int(px), int(py)))
                # Small dot
                pygame.draw.circle(self.screen, (59, 130, 246, 100), (int(px), int(py)), 2)

        # Draw mean line
        if len(display_means) > 1:
            step = chart_w / max(1, len(display_means) - 1)
            mean_points = []
            for i, m in enumerate(display_means):
                px = chart_x + i * step
                py = chart_y + chart_h - (m / max_score) * chart_h
                mean_points.append((int(px), int(py)))
            if len(mean_points) >= 2:
                pygame.draw.lines(self.screen, NEON_GREEN, False, mean_points, 2)

    def get_window_size(self):
        """Return current window dimensions."""
        return self.window_width, self.window_height
