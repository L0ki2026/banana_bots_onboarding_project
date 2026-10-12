"""Tests for navigation environment, pathfinding, and reward shaping."""

from __future__ import annotations

import numpy as np
import pytest

from banana_bots_rl import config
from banana_bots_rl.diagnostics import count_two_cell_oscillations
from banana_bots_rl.environment import AssistiveNavigationEnv, DIR_DELTA, STRAFE_LEFT, STRAFE_RIGHT
from banana_bots_rl.map_generator import MapLayout
from banana_bots_rl.pathfinding import distance_field_to_goal, shortest_path_distance
from banana_bots_rl.reward_shaping import PotentialBasedRewardShaping


def _empty_layout(
    size: int = 8,
    start: tuple[int, int] = (2, 2),
    goal: tuple[int, int] = (5, 5),
) -> MapLayout:
    grid = np.zeros((size, size), dtype=np.int32)
    grid[goal[0], goal[1]] = config.GOAL
    return MapLayout(grid=grid, start=start, goal=goal)


@pytest.fixture
def env() -> AssistiveNavigationEnv:
    return AssistiveNavigationEnv(
        grid_size=12,
        view_distance=3,
        procedural=False,
        max_steps=50,
    )


@pytest.mark.parametrize(
    "facing,action,expected_delta",
    [
        (0, 0, DIR_DELTA[0]),
        (0, 1, (-DIR_DELTA[0][0], -DIR_DELTA[0][1])),
        (0, 2, STRAFE_LEFT[0]),
        (0, 3, STRAFE_RIGHT[0]),
        (1, 0, DIR_DELTA[1]),
        (2, 3, STRAFE_RIGHT[2]),
        (3, 2, STRAFE_LEFT[3]),
    ],
)
def test_movement_actions(
    env: AssistiveNavigationEnv,
    facing: int,
    action: int,
    expected_delta: tuple[int, int],
) -> None:
    layout = _empty_layout()
    env.reset(options={"layout": layout})
    env.facing = facing
    r0, c0 = env.agent_pos
    env.step(action)
    assert env.agent_pos == (r0 + expected_delta[0], c0 + expected_delta[1])


def test_collision_keeps_position(env: AssistiveNavigationEnv) -> None:
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[2, 3] = config.OBSTACLE
    grid[4, 4] = config.GOAL
    layout = MapLayout(grid=grid, start=(2, 2), goal=(4, 4))
    env.reset(options={"layout": layout})
    env.facing = 1  # East into obstacle
    _, reward, term, trunc, info = env.step(0)
    assert env.agent_pos == (2, 2)
    assert info["collision"] is True
    assert reward <= config.STEP_PENALTY + config.COLLISION_PENALTY
    assert not term and not trunc


def test_observation_updates_after_move(env: AssistiveNavigationEnv) -> None:
    layout = _empty_layout(start=(3, 3), goal=(3, 7))
    env.reset(options={"layout": layout})
    env.facing = 1
    obs_before = env._get_obs().copy()
    env.step(0)  # forward East
    obs_after = env._get_obs()
    assert not np.allclose(obs_before, obs_after)


def test_observation_consistent_with_state(env: AssistiveNavigationEnv) -> None:
    layout = _empty_layout()
    env.reset(options={"layout": layout})
    env.facing = 2
    env.step(0)
    assert np.allclose(env._get_obs(), env._build_observation())


def test_bfs_distance_with_obstacle() -> None:
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[2, 1] = config.OBSTACLE
    grid[2, 2] = config.OBSTACLE
    grid[2, 3] = config.OBSTACLE
    start, goal = (2, 0), (2, 4)
    d = shortest_path_distance(grid, start, goal)
    assert d == 6  # detour above/below the wall


def test_distance_field_matches_point_queries() -> None:
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[1, 2] = config.OBSTACLE
    goal = (5, 5)
    grid[goal[0], goal[1]] = config.GOAL
    field = distance_field_to_goal(grid, goal)
    start = (0, 0)
    assert int(field[start[0], start[1]]) == shortest_path_distance(grid, start, goal)


def test_reward_shaping_progress_vs_regression(env: AssistiveNavigationEnv) -> None:
    layout = _empty_layout(start=(2, 2), goal=(2, 6))
    shaped = PotentialBasedRewardShaping(env, beta=1.0, gamma=config.PPO_GAMMA, enabled=True)
    shaped.reset(options={"layout": layout})
    env.facing = 1
    _, r_forward, _, _, info_f = shaped.step(0)
    shaped.reset(options={"layout": layout})
    env.facing = 3
    _, r_back, _, _, info_b = shaped.step(0)
    assert info_f["reward_shaping_bonus"] > info_b["reward_shaping_bonus"]


def test_reward_shaping_no_bonus_on_collision(env: AssistiveNavigationEnv) -> None:
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[2, 3] = config.OBSTACLE
    grid[3, 3] = config.GOAL
    layout = MapLayout(grid=grid, start=(2, 2), goal=(3, 3))
    shaped = PotentialBasedRewardShaping(env, enabled=True)
    shaped.reset(options={"layout": layout})
    env.facing = 1
    _, _, _, _, info = shaped.step(0)
    assert info["collision"] is True
    assert info["reward_shaping_bonus"] == 0.0


def test_termination_on_goal(env: AssistiveNavigationEnv) -> None:
    grid = np.zeros((4, 4), dtype=np.int32)
    start, goal = (2, 2), (2, 3)
    grid[goal[0], goal[1]] = config.GOAL
    layout = MapLayout(grid=grid, start=start, goal=goal)
    env.reset(options={"layout": layout})
    env.facing = 1
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert reward >= config.GOAL_REWARD + config.STEP_PENALTY


def test_truncation_without_goal(env: AssistiveNavigationEnv) -> None:
    layout = _empty_layout(start=(0, 0), goal=(7, 7))
    env.max_steps = 3
    env.reset(options={"layout": layout})
    terminated = truncated = False
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(0)
    assert terminated is False
    assert truncated is True
    assert info["success"] is False


def test_two_cell_oscillation_detection() -> None:
    positions = [(7, 7), (8, 7), (7, 7), (8, 7), (7, 7)]
    assert count_two_cell_oscillations(positions) == 3


def test_agent_frame_goal_direction(env: AssistiveNavigationEnv) -> None:
    layout = _empty_layout(start=(2, 2), goal=(2, 6))
    env.reset(options={"layout": layout})
    env.facing = 1  # East
    vec = env._goal_direction_agent_frame()
    assert vec[0] > 0.9  # goal is ahead
    assert abs(vec[1]) < 0.1


def test_curriculum_map_generators() -> None:
    from banana_bots_rl.map_generator import generate_map_for_difficulty

    rng = np.random.default_rng(0)
    easy = generate_map_for_difficulty("easy", size=12, rng=rng)
    assert np.sum(easy.grid == config.OBSTACLE) == 0
    medium = generate_map_for_difficulty("medium", size=12, rng=rng)
    assert medium.grid.shape == (12, 12)


def test_env_tracks_two_cell_oscillation(env: AssistiveNavigationEnv) -> None:
    layout = _empty_layout(start=(7, 7), goal=(0, 0))
    env.reset(options={"layout": layout})
    env.facing = 1
    for action in (3, 2, 3, 2, 3):
        _, _, _, _, info = env.step(action)
    assert info["two_cell_oscillations"] >= 2
