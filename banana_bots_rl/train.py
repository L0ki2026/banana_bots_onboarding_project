"""PPO training with Stable-Baselines3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from banana_bots_rl import config
from banana_bots_rl.curriculum import CurriculumCallback
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.map_generator import Difficulty
from banana_bots_rl.model_utils import ensure_save_path
from banana_bots_rl.reward_shaping import PotentialBasedRewardShaping
from banana_bots_rl.utils import rolling_mean

TrainingMode = Literal["baseline", "shaped", "curriculum"]


class EpisodeMetricsCallback(BaseCallback):
    """Log success, length, collisions, and oscillations for rolling plots."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_success: list[float] = []
        self.episode_collisions: list[float] = []
        self.episode_lengths: list[float] = []
        self.episode_oscillations: list[float] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(float(ep["l"]))
            if "success" in info:
                self.episode_success.append(float(info["success"]))
                self.episode_collisions.append(float(info.get("collisions_total", 0)))
                self.episode_oscillations.append(
                    float(info.get("two_cell_oscillations", 0))
                )
        return True


def make_train_env(
    seed: int,
    *,
    reward_shaping: bool = False,
    shaping_beta: float = config.REWARD_SHAPING_BETA,
    difficulty: Difficulty = "hard",
) -> gym.Env:
    env = AssistiveNavigationEnv(
        procedural=True,
        render_mode=None,
        difficulty=difficulty,
    )
    if reward_shaping:
        env = PotentialBasedRewardShaping(
            env,
            beta=shaping_beta,
            gamma=config.PPO_GAMMA,
            enabled=True,
        )
    env = Monitor(
        env,
        info_keywords=(
            "success",
            "collisions_total",
            "two_cell_oscillations",
        ),
    )
    env.reset(seed=seed)
    return env


def build_model(env: gym.Env, seed: int = config.RANDOM_SEED) -> PPO:
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
    )
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.PPO_LEARNING_RATE,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH_SIZE,
        gamma=config.PPO_GAMMA,
        ent_coef=config.PPO_ENT_COEF,
        verbose=1,
        seed=seed,
    )


def default_save_path(mode: TrainingMode, reward_shaping: bool) -> str:
    if mode == "curriculum":
        return config.MODEL_SAVE_PATH_CURRICULUM
    if mode == "shaped" or reward_shaping:
        return config.MODEL_SAVE_PATH_SHAPED
    return config.MODEL_SAVE_PATH_BASELINE


def plot_training_rewards(
    rewards: list[float],
    window: int = 50,
    save_path: Path | None = None,
) -> None:
    if not rewards:
        return
    x = np.arange(len(rewards))
    smoothed = rolling_mean(rewards, window)
    plt.figure(figsize=(8, 4))
    plt.plot(x, rewards, alpha=0.25, label="episode reward")
    plt.plot(x, smoothed, label=f"rolling mean ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training reward")
    plt.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def train(
    total_timesteps: int = config.TOTAL_TIMESTEPS,
    save_path: str = config.MODEL_SAVE_PATH_BASELINE,
    plot_path: str = "plots/training_reward.png",
    seed: int = config.RANDOM_SEED,
    reward_shaping: bool = False,
    shaping_beta: float = config.REWARD_SHAPING_BETA,
    mode: TrainingMode = "baseline",
    overwrite: bool = False,
) -> PPO:
    use_curriculum = mode == "curriculum"
    start_difficulty: Difficulty = (
        config.CURRICULUM_START_DIFFICULTY if use_curriculum else "hard"
    )
    if use_curriculum and not reward_shaping:
        reward_shaping = True

    env = make_train_env(
        seed,
        reward_shaping=reward_shaping,
        shaping_beta=shaping_beta,
        difficulty=start_difficulty,
    )

    try:
        from gymnasium.utils.env_checker import check_env

        check_env(
            AssistiveNavigationEnv(procedural=True, difficulty="easy"),
            skip_render_check=True,
        )
    except Exception as e:
        print(f"Env checker note: {e}")

    save_file = ensure_save_path(save_path, overwrite=overwrite)

    model = build_model(env, seed=seed)
    metrics_cb = EpisodeMetricsCallback()
    callbacks: list[BaseCallback] = [metrics_cb]
    if use_curriculum:
        callbacks.append(CurriculumCallback(env))

    print(
        f"Training PPO mode={mode} for {total_timesteps} timesteps "
        f"(seed={seed}, reward_shaping={reward_shaping}, beta={shaping_beta}, "
        f"start_difficulty={start_difficulty})"
    )
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    model.save(str(save_file))
    print(f"Saved model to {save_file}")

    rewards = metrics_cb.episode_rewards
    if rewards:
        print(
            f"Episodes: {len(rewards)}, "
            f"mean success: {np.mean(metrics_cb.episode_success):.1%}, "
            f"mean length: {np.mean(metrics_cb.episode_lengths):.1f}, "
            f"mean 2-cell oscillations: {np.mean(metrics_cb.episode_oscillations):.2f}"
        )
    plot_training_rewards(rewards, save_path=Path(plot_path) if plot_path else None)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO navigation policy")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--mode",
        choices=("baseline", "shaped", "curriculum"),
        default="baseline",
        help="baseline=hard maps; shaped=hard+PBRS; curriculum=easy->hard+PBRS",
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Enable potential-based reward shaping (default for shaped/curriculum modes)",
    )
    parser.add_argument(
        "--shaping-beta",
        type=float,
        default=config.REWARD_SHAPING_BETA,
        help="Scaling for potential-based shaping term",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="plots/training_reward.png",
        help="Path for training reward plot",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing checkpoint file",
    )
    args = parser.parse_args()

    reward_shaping = args.reward_shaping or args.mode in ("shaped", "curriculum")
    save_path = args.save or default_save_path(args.mode, reward_shaping)

    train(
        total_timesteps=args.timesteps,
        save_path=save_path,
        plot_path=args.plot,
        seed=args.seed,
        reward_shaping=reward_shaping,
        shaping_beta=args.shaping_beta,
        mode=args.mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
