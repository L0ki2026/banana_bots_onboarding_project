"""Pygame rendering and Colab-friendly rgb_array frames."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from banana_bots_rl import config

# Colors (RGB)
COLOR_EMPTY = (245, 245, 245)
COLOR_GRASS = (120, 180, 90)
COLOR_OBSTACLE = (60, 60, 60)
COLOR_GOAL = (220, 50, 50)
COLOR_AGENT = (50, 100, 220)


def render_grid_to_array(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    facing: int,
    cell_size: int = config.CELL_PIXEL_SIZE,
) -> np.ndarray:
    """Headless RGB image of the grid (no pygame display required)."""
    h, w = grid.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    for r in range(h):
        for c in range(w):
            cell = grid[r, c]
            if cell == config.GRASS:
                color = COLOR_GRASS
            elif cell == config.OBSTACLE:
                color = COLOR_OBSTACLE
            elif cell == config.GOAL:
                color = COLOR_GOAL
            else:
                color = COLOR_EMPTY
            y0, x0 = r * cell_size, c * cell_size
            img[y0 : y0 + cell_size, x0 : x0 + cell_size] = color

    ar, ac = agent_pos
    y0, x0 = ar * cell_size, ac * cell_size
    img[y0 : y0 + cell_size, x0 : x0 + cell_size] = COLOR_AGENT
    _draw_facing_triangle(img, ar, ac, facing, cell_size)

    return img


def _draw_facing_triangle(
    img: np.ndarray, r: int, c: int, facing: int, cell_size: int
) -> None:
    """Small white wedge indicating facing direction."""
    cy = r * cell_size + cell_size // 2
    cx = c * cell_size + cell_size // 2
    d = cell_size // 4
    if facing == 0:
        pts = [(cx, cy - d), (cx - d, cy + d), (cx + d, cy + d)]
    elif facing == 1:
        pts = [(cx + d, cy), (cx - d, cy - d), (cx - d, cy + d)]
    elif facing == 2:
        pts = [(cx, cy + d), (cx - d, cy - d), (cx + d, cy - d)]
    else:
        pts = [(cx - d, cy), (cx + d, cy - d), (cx + d, cy + d)]

    for py, px in pts:
        if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
            img[py, px] = (255, 255, 255)
    # Fill center toward facing edge
    cy = r * cell_size + cell_size // 2
    cx = c * cell_size + cell_size // 2
    if facing == 0:
        img[cy - d : cy, cx - 1 : cx + 2] = (255, 255, 255)
    elif facing == 1:
        img[cy - 1 : cy + 2, cx : cx + d] = (255, 255, 255)
    elif facing == 2:
        img[cy : cy + d, cx - 1 : cx + 2] = (255, 255, 255)
    else:
        img[cy - 1 : cy + 2, cx - d : cx] = (255, 255, 255)


def play_episode_pygame(
    env,
    model=None,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Step through an episode with pygame window if display available.
    Returns list of rgb frames either way.
    """
    import pygame

    frames: List[np.ndarray] = []
    obs, _ = env.reset()
    steps = 0
    limit = max_steps or config.MAX_STEPS

    pygame.init()
    use_display = os.environ.get("COLAB_RELEASE_TAG") is None
    screen = None
    clock = pygame.time.Clock()

    while steps < limit:
        frame = env._render_frame()
        frames.append(frame.copy())
        if use_display:
            if screen is None:
                h, w = frame.shape[:2]
                screen = pygame.display.set_mode((w, h))
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return frames
            clock.tick(config.FPS)

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, _, terminated, truncated, _ = env.step(int(action))
        steps += 1
        if terminated or truncated:
            frames.append(env._render_frame().copy())
            break

    pygame.quit()
    return frames


def show_frames_in_notebook(frames: List[np.ndarray], interval_ms: int = 200) -> None:
    """Display animation in IPython / Colab."""
    try:
        from matplotlib import animation
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for notebook display")

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_array(frames[i % len(frames)])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval_ms, repeat=False
    )
    from IPython.display import HTML

    plt.close(fig)
    return HTML(anim.to_jshtml())
