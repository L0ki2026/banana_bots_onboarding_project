"""Checkpoint loading and compatibility checks."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def validate_observation_space(model: PPO, env: gym.Env) -> None:
    """Raise if the checkpoint does not match the environment observation space."""
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            "Checkpoint observation space "
            f"{model.observation_space.shape} does not match environment "
            f"{env.observation_space.shape}. Retrain with the current environment."
        )
    if model.action_space.n != env.action_space.n:
        raise ValueError(
            f"Checkpoint action space n={model.action_space.n} != env n={env.action_space.n}"
        )


def load_ppo_for_env(path: str | Path, env: gym.Env) -> PPO:
    """Load a PPO checkpoint after verifying it matches the environment."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    model = PPO.load(str(path))
    validate_observation_space(model, env)
    return model


def ensure_save_path(path: str | Path, *, overwrite: bool = False) -> Path:
    """Return path for saving; refuse to clobber an existing checkpoint by default."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {path}. "
            "Pass --overwrite or choose a different --save path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
