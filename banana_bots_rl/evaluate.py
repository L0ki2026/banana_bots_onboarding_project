"""Held-out map evaluation and random baseline comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from banana_bots_rl import config
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.map_generator import MapLayout, generate_eval_maps


@dataclass
class EvalSummary:
    name: str
    n_episodes: int
    success_rate: float
    collision_rate: float  # fraction of episodes with >=1 collision
    avg_collisions: float
    avg_steps_success: float
    avg_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_episode_on_layout(
    layout: MapLayout,
    policy: Callable[[np.ndarray, AssistiveNavigationEnv], int],
    seed: int = 0,
) -> Dict[str, Any]:
    env = AssistiveNavigationEnv(procedural=False, render_mode=None)
    obs, _ = env.reset(seed=seed, options={"layout": layout})
    total_reward = 0.0
    done = False
    while not done:
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "success": info.get("success", False),
        "collisions": info.get("collisions_total", 0),
        "steps": info.get("steps", 0),
        "reward": total_reward,
    }


def random_policy(obs: np.ndarray, env: AssistiveNavigationEnv) -> int:
    return int(env.action_space.sample())


def sb3_policy(model, deterministic: bool = True):
    def _policy(obs: np.ndarray, env: AssistiveNavigationEnv) -> int:
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    return _policy


def evaluate_on_maps(
    layouts: List[MapLayout],
    policy: Callable[[np.ndarray, AssistiveNavigationEnv], int],
    name: str = "agent",
) -> EvalSummary:
    results = []
    for i, layout in enumerate(layouts):
        results.append(run_episode_on_layout(layout, policy, seed=i))

    n = len(results)
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / n if n else 0.0
    collision_episodes = sum(1 for r in results if r["collisions"] > 0)
    avg_collisions = float(np.mean([r["collisions"] for r in results]))
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_steps_success = (
        float(np.mean([r["steps"] for r in successes])) if successes else float("nan")
    )

    return EvalSummary(
        name=name,
        n_episodes=n,
        success_rate=success_rate,
        collision_rate=collision_episodes / n if n else 0.0,
        avg_collisions=avg_collisions,
        avg_steps_success=avg_steps_success,
        avg_reward=avg_reward,
    )


def load_or_create_eval_maps(path: Optional[Path] = None) -> List[MapLayout]:
    """Generate held-out maps (fixed eval seed)."""
    return generate_eval_maps()


def compare_agents(
    model=None,
    n_maps: int = config.NUM_EVAL_MAPS,
    save_path: Optional[Path] = None,
) -> Dict[str, EvalSummary]:
    layouts = generate_eval_maps(n=n_maps)
    summaries = {
        "random": evaluate_on_maps(layouts, random_policy, name="random"),
    }
    if model is not None:
        summaries["ppo"] = evaluate_on_maps(
            layouts, sb3_policy(model), name="ppo"
        )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in summaries.items()}, f, indent=2)

    return summaries


def print_summary(summary: EvalSummary) -> None:
    print(f"--- {summary.name} ---")
    print(f"  Success rate:      {summary.success_rate:.1%}")
    print(f"  Collision rate:    {summary.collision_rate:.1%} (episodes with collision)")
    print(f"  Avg collisions:    {summary.avg_collisions:.2f}")
    print(f"  Avg steps (success): {summary.avg_steps_success:.1f}")
    print(f"  Avg reward:        {summary.avg_reward:.2f}")


if __name__ == "__main__":
    layouts = load_or_create_eval_maps()
    s = evaluate_on_maps(layouts[:10], random_policy, name="random")
    print_summary(s)
