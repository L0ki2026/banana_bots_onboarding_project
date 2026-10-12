"""Pygame rendering and Colab-friendly rgb_array frames."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from banana_bots_rl import config
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.map_generator import MapLayout, generate_eval_maps
from banana_bots_rl.model_utils import load_ppo_for_env

# Colors (RGB)
COLOR_EMPTY = (245, 245, 245)
COLOR_GRASS = (120, 180, 90)
COLOR_OBSTACLE = (60, 60, 60)
COLOR_GOAL = (220, 50, 50)
COLOR_AGENT = (50, 100, 220)
COLOR_PATH = (255, 180, 60)


@dataclass
class EpisodePlaybackResult:
    frames: List[np.ndarray]
    success: bool
    truncated: bool
    steps: int
    total_reward: float
    collisions: int
    two_cell_oscillations: int
    positions: List[Tuple[int, int]]

    @property
    def outcome_label(self) -> str:
        if self.success:
            return "SUCCESS (goal reached)"
        if self.truncated:
            return "TIMEOUT (max steps)"
        return "ENDED"


def render_grid_to_array(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    facing: int,
    cell_size: int = config.CELL_PIXEL_SIZE,
    path_positions: Optional[Sequence[Tuple[int, int]]] = None,
    hud_lines: Optional[Sequence[str]] = None,
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

    if path_positions:
        _draw_path(img, path_positions, cell_size)

    ar, ac = agent_pos
    y0, x0 = ar * cell_size, ac * cell_size
    img[y0 : y0 + cell_size, x0 : x0 + cell_size] = COLOR_AGENT
    _draw_facing_triangle(img, ar, ac, facing, cell_size)

    if hud_lines:
        _draw_hud(img, hud_lines)

    return img


def _draw_path(
    img: np.ndarray, positions: Sequence[Tuple[int, int]], cell_size: int
) -> None:
    for r, c in positions:
        cy = r * cell_size + cell_size // 2
        cx = c * cell_size + cell_size // 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                py, px = cy + dy, cx + dx
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    img[py, px] = COLOR_PATH


def _draw_hud(img: np.ndarray, lines: Sequence[str]) -> None:
    """Top banner with text when pygame font is available."""
    bar_h = max(18, 14 + 12 * len(lines))
    img[:bar_h, :, :] = (30, 30, 30)
    try:
        import pygame

        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont("dejavusans", 14)
        y = 4
        for line in lines:
            surf = font.render(line, True, (240, 240, 240))
            arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
            h, w = arr.shape[:2]
            if y + h <= bar_h and w <= img.shape[1]:
                img[y : y + h, :w, :] = arr
            y += h
    except (ImportError, OSError, RuntimeError):
        pass


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
    env: AssistiveNavigationEnv,
    model=None,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    layout: Optional[MapLayout] = None,
    draw_path: bool = True,
    use_display: Optional[bool] = None,
) -> EpisodePlaybackResult:
    """
    Step through an episode; capture rgb frames (optional pygame window locally).
    """
    reset_options: dict[str, Any] = {}
    if layout is not None:
        reset_options["layout"] = layout
    obs, _ = env.reset(seed=seed, options=reset_options or None)

    steps = 0
    limit = max_steps or env.max_steps
    total_reward = 0.0
    positions: List[Tuple[int, int]] = [env.agent_pos]
    frames: List[np.ndarray] = []
    terminated = truncated = False
    last_info: dict = {}

    if use_display is None:
        use_display = os.environ.get("COLAB_RELEASE_TAG") is None

    pygame_module = None
    screen = None
    clock = None
    if use_display:
        import pygame

        pygame_module = pygame
        pygame.init()
        clock = pygame.time.Clock()

    def capture_frame() -> None:
        path = positions if draw_path else None
        frame = render_grid_to_array(
            env.grid,
            env.agent_pos,
            env.goal_pos,
            env.facing,
            path_positions=path,
            hud_lines=None,
        )
        frames.append(frame.copy())
        if use_display and pygame_module is not None:
            nonlocal screen
            if screen is None:
                h, w = frame.shape[:2]
                screen = pygame_module.display.set_mode((w, h))
            surf = pygame_module.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame_module.display.flip()
            for event in pygame_module.event.get():
                if event.type == pygame_module.QUIT:
                    return

    capture_frame()

    while steps < limit and not (terminated or truncated):
        if model is None:
            action = int(env.action_space.sample())
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        steps += 1
        positions.append(env.agent_pos)
        capture_frame()

        if use_display and clock is not None:
            clock.tick(config.FPS)

    if not frames or positions[-1] != env.agent_pos:
        capture_frame()

    if use_display and pygame_module is not None:
        pygame_module.quit()

    success = bool(terminated and last_info.get("success", False))
    outcome = (
        "SUCCESS (goal reached)"
        if success
        else ("TIMEOUT" if truncated else "ENDED")
    )
    if frames:
        hud = [
            outcome,
            f"steps={last_info.get('steps', steps)} reward={total_reward:.1f}",
            f"collisions={last_info.get('collisions_total', 0)} "
            f"oscillations={last_info.get('two_cell_oscillations', 0)}",
        ]
        frames[-1] = render_grid_to_array(
            env.grid,
            env.agent_pos,
            env.goal_pos,
            env.facing,
            path_positions=positions if draw_path else None,
            hud_lines=hud,
        )

    return EpisodePlaybackResult(
        frames=frames,
        success=success,
        truncated=bool(truncated and not success),
        steps=last_info.get("steps", steps),
        total_reward=total_reward,
        collisions=int(last_info.get("collisions_total", 0)),
        two_cell_oscillations=int(last_info.get("two_cell_oscillations", 0)),
        positions=positions,
    )


def show_frames_in_notebook(frames: List[np.ndarray], interval_ms: int = 200) -> None:
    """Display animation in IPython / Colab."""
    try:
        from matplotlib import animation
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for notebook display")

    if not frames:
        raise ValueError("No frames to display")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a navigation episode")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="PPO checkpoint compatible with AssistiveNavigationEnv",
    )
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--eval-map-index", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    args = parser.parse_args()

    layouts = generate_eval_maps(n=max(1, args.eval_map_index + 1))
    layout = layouts[args.eval_map_index]
    env = AssistiveNavigationEnv(procedural=False, render_mode="rgb_array")
    model = load_ppo_for_env(args.model_path, env)
    result = play_episode_pygame(
        env,
        model=model,
        deterministic=args.deterministic,
        seed=args.seed,
        layout=layout,
        use_display=False,
    )
    print(result.outcome_label)
    print(f"Steps: {result.steps}, reward: {result.total_reward:.2f}")
    print(f"Collisions: {result.collisions}, 2-cell oscillations: {result.two_cell_oscillations}")


if __name__ == "__main__":
    main()
