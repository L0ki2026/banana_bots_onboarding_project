"""Held-out map evaluation and random baseline comparison."""

from __future__ import annotations

import argparse
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
    avg_steps: float
    avg_steps_success: float
    avg_reward: float
    avg_two_cell_oscillations: float
    oscillation_episode_rate: float  # episodes with >=1 two-cell oscillation

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
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    success = bool(terminated and info.get("success", False))
    return {
        "success": success,
        "collisions": info.get("collisions_total", 0),
        "steps": info.get("steps", 0),
        "reward": total_reward,
        "two_cell_oscillations": info.get("two_cell_oscillations", 0),
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
    eval_seeds: Optional[List[int]] = None,
) -> EvalSummary:
    results = []
    for i, layout in enumerate(layouts):
        seed = eval_seeds[i] if eval_seeds is not None else i
        results.append(run_episode_on_layout(layout, policy, seed=seed))

    n = len(results)
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / n if n else 0.0
    collision_episodes = sum(1 for r in results if r["collisions"] > 0)
    avg_collisions = float(np.mean([r["collisions"] for r in results])) if n else 0.0
    avg_reward = float(np.mean([r["reward"] for r in results])) if n else 0.0
    avg_steps = float(np.mean([r["steps"] for r in results])) if n else 0.0
    avg_steps_success = (
        float(np.mean([r["steps"] for r in successes])) if successes else float("nan")
    )
    avg_osc = float(np.mean([r["two_cell_oscillations"] for r in results])) if n else 0.0
    osc_eps = sum(1 for r in results if r["two_cell_oscillations"] > 0)
    osc_rate = osc_eps / n if n else 0.0

    return EvalSummary(
        name=name,
        n_episodes=n,
        success_rate=success_rate,
        collision_rate=collision_episodes / n if n else 0.0,
        avg_collisions=avg_collisions,
        avg_steps=avg_steps,
        avg_steps_success=avg_steps_success,
        avg_reward=avg_reward,
        avg_two_cell_oscillations=avg_osc,
        oscillation_episode_rate=osc_rate,
    )


def load_or_create_eval_maps(
    n: int = config.NUM_EVAL_MAPS,
    seed: int = config.EVAL_SEED,
) -> List[MapLayout]:
    """Generate held-out maps (fixed eval seed)."""
    return generate_eval_maps(n=n, seed=seed)


def eval_seeds_for_maps(n: int, base: int = config.EVAL_SEED) -> List[int]:
    """Per-episode reset seeds (deterministic, identical across agent comparisons)."""
    rng = np.random.default_rng(base + 999)
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]


def compare_agents(
    model=None,
    model_original=None,
    model_shaped=None,
    n_maps: int = config.NUM_EVAL_MAPS,
    save_path: Optional[Path] = None,
) -> Dict[str, EvalSummary]:
    if model_original is None and model is not None:
        model_original = model
    layouts = load_or_create_eval_maps(n=n_maps)
    seeds = eval_seeds_for_maps(n_maps)
    summaries: Dict[str, EvalSummary] = {
        "random": evaluate_on_maps(
            layouts, random_policy, name="random", eval_seeds=seeds
        ),
    }
    if model_original is not None:
        summaries["ppo_original_deterministic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model_original, deterministic=True),
            name="ppo_original_deterministic",
            eval_seeds=seeds,
        )
        summaries["ppo_original_stochastic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model_original, deterministic=False),
            name="ppo_original_stochastic",
            eval_seeds=seeds,
        )
    if model_shaped is not None:
        summaries["ppo_shaped_deterministic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model_shaped, deterministic=True),
            name="ppo_shaped_deterministic",
            eval_seeds=seeds,
        )
        summaries["ppo_shaped_stochastic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model_shaped, deterministic=False),
            name="ppo_shaped_stochastic",
            eval_seeds=seeds,
        )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in summaries.items()}, f, indent=2)

    return summaries


def print_summary(summary: EvalSummary) -> None:
    print(f"--- {summary.name} ---")
    print(f"  Success rate:           {summary.success_rate:.1%}")
    print(f"  Collision rate:         {summary.collision_rate:.1%} (episodes with collision)")
    print(f"  Avg collisions:         {summary.avg_collisions:.2f}")
    print(f"  Avg steps:              {summary.avg_steps:.1f}")
    print(f"  Avg steps (success):    {summary.avg_steps_success:.1f}")
    print(f"  Avg reward:             {summary.avg_reward:.2f}")
    print(f"  Avg 2-cell oscillations:{summary.avg_two_cell_oscillations:.2f}")
    print(f"  Oscillation ep. rate:   {summary.oscillation_episode_rate:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate navigation agents on held-out maps")
    parser.add_argument(
        "--model-original",
        type=str,
        default=config.MODEL_SAVE_PATH,
        help="Path to original PPO checkpoint",
    )
    parser.add_argument(
        "--model-shaped",
        type=str,
        default=None,
        help="Path to reward-shaped PPO checkpoint (optional)",
    )
    parser.add_argument("--n-maps", type=int, default=config.NUM_EVAL_MAPS)
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_comparison.json",
        help="JSON file for metrics",
    )
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Only run random baseline (no model files required)",
    )
    args = parser.parse_args()

    model_original = None
    model_shaped = None
    if not args.skip_ppo:
        from stable_baselines3 import PPO

        orig_path = Path(args.model_original)
        if orig_path.exists():
            model_original = PPO.load(str(orig_path))
        else:
            print(f"Warning: original model not found at {orig_path}")
        if args.model_shaped:
            shaped_path = Path(args.model_shaped)
            if shaped_path.exists():
                model_shaped = PPO.load(str(shaped_path))
            else:
                print(f"Warning: shaped model not found at {shaped_path}")

    summaries = compare_agents(
        model_original=model_original,
        model_shaped=model_shaped,
        n_maps=args.n_maps,
        save_path=Path(args.output),
    )
    for summary in summaries.values():
        print_summary(summary)


if __name__ == "__main__":
    main()
