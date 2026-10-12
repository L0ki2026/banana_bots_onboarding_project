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
from banana_bots_rl.model_utils import load_ppo_for_env, validate_observation_space


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


def evaluate_model_paths(
    model_paths: List[Path],
    n_maps: int = config.NUM_EVAL_MAPS,
) -> Dict[str, EvalSummary]:
    layouts = load_or_create_eval_maps(n=n_maps)
    seeds = eval_seeds_for_maps(n_maps)
    probe_env = AssistiveNavigationEnv(procedural=False, render_mode=None)

    summaries: Dict[str, EvalSummary] = {
        "random": evaluate_on_maps(
            layouts, random_policy, name="random", eval_seeds=seeds
        ),
    }

    for path in model_paths:
        model = load_ppo_for_env(path, probe_env)
        stem = path.stem
        summaries[f"{stem}_deterministic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model, deterministic=True),
            name=f"{stem}_deterministic",
            eval_seeds=seeds,
        )
        summaries[f"{stem}_stochastic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model, deterministic=False),
            name=f"{stem}_stochastic",
            eval_seeds=seeds,
        )
    return summaries


def compare_agents(
    model=None,
    model_original=None,
    model_shaped=None,
    n_maps: int = config.NUM_EVAL_MAPS,
    save_path: Optional[Path] = None,
    extra_model_paths: Optional[List[str]] = None,
) -> Dict[str, EvalSummary]:
    paths: List[Path] = []
    if extra_model_paths:
        paths.extend(Path(p) for p in extra_model_paths)
    if model_original is not None or model is not None:
        pass  # legacy in-memory models handled below

    layouts = load_or_create_eval_maps(n=n_maps)
    seeds = eval_seeds_for_maps(n_maps)
    summaries: Dict[str, EvalSummary] = {
        "random": evaluate_on_maps(
            layouts, random_policy, name="random", eval_seeds=seeds
        ),
    }

    probe_env = AssistiveNavigationEnv(procedural=False, render_mode=None)

    def add_model(model_obj, prefix: str) -> None:
        validate_observation_space(model_obj, probe_env)
        summaries[f"{prefix}_deterministic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model_obj, deterministic=True),
            name=f"{prefix}_deterministic",
            eval_seeds=seeds,
        )
        summaries[f"{prefix}_stochastic"] = evaluate_on_maps(
            layouts,
            sb3_policy(model_obj, deterministic=False),
            name=f"{prefix}_stochastic",
            eval_seeds=seeds,
        )

    if model_original is None and model is not None:
        model_original = model
    if model_original is not None:
        add_model(model_original, "ppo_original")
    if model_shaped is not None:
        add_model(model_shaped, "ppo_shaped")

    if paths:
        for path in paths:
            if not path.exists():
                print(f"Warning: model not found at {path}")
                continue
            m = load_ppo_for_env(path, probe_env)
            add_model(m, path.stem)

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
        "--model-path",
        action="append",
        default=[],
        help="PPO checkpoint (.zip); may be repeated for multiple models",
    )
    parser.add_argument(
        "--model-original",
        type=str,
        default=None,
        help="Legacy alias for a single checkpoint path",
    )
    parser.add_argument(
        "--model-shaped",
        type=str,
        default=None,
        help="Optional second checkpoint (legacy compare)",
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

    paths = [Path(p) for p in args.model_path]
    if args.model_original:
        paths.append(Path(args.model_original))
    if args.model_shaped:
        paths.append(Path(args.model_shaped))

    if args.skip_ppo or not paths:
        layouts = load_or_create_eval_maps(n=args.n_maps)
        seeds = eval_seeds_for_maps(args.n_maps)
        summaries = {
            "random": evaluate_on_maps(
                layouts, random_policy, name="random", eval_seeds=seeds
            )
        }
    else:
        summaries = evaluate_model_paths(paths, n_maps=args.n_maps)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({k: v.to_dict() for k, v in summaries.items()}, f, indent=2)

    for summary in summaries.values():
        print_summary(summary)


if __name__ == "__main__":
    main()
