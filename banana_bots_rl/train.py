"""PPO training with Stable-Baselines3."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from banana_bots_rl import config
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.reward_shaping import PotentialBasedRewardShaping
from banana_bots_rl.utils import rolling_mean


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
) -> gym.Env:
    env = AssistiveNavigationEnv(procedural=True, render_mode=None)
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
        verbose=1,
        seed=seed,
    )


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
    save_path: str = config.MODEL_SAVE_PATH,
    plot_path: str = "plots/training_reward.png",
    seed: int = config.RANDOM_SEED,
    reward_shaping: bool = False,
    shaping_beta: float = config.REWARD_SHAPING_BETA,
) -> PPO:
    env = make_train_env(seed, reward_shaping=reward_shaping, shaping_beta=shaping_beta)

    try:
        from gymnasium.utils.env_checker import check_env

        check_env(AssistiveNavigationEnv(procedural=True), skip_render_check=True)
    except Exception as e:
        print(f"Env checker note: {e}")

    model = build_model(env, seed=seed)
    callback = EpisodeMetricsCallback()

    print(
        f"Training PPO for {total_timesteps} timesteps "
        f"(seed={seed}, reward_shaping={reward_shaping}, beta={shaping_beta})"
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Saved model to {save_path}")

    rewards = callback.episode_rewards
    if rewards:
        print(
            f"Episodes: {len(rewards)}, "
            f"mean success: {np.mean(callback.episode_success):.1%}, "
            f"mean length: {np.mean(callback.episode_lengths):.1f}, "
            f"mean 2-cell oscillations: {np.mean(callback.episode_oscillations):.2f}"
        )
    plot_training_rewards(rewards, save_path=Path(plot_path) if plot_path else None)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO navigation policy")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Enable potential-based reward shaping during training",
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
    args = parser.parse_args()

    if args.save is None:
        save_path = (
            config.MODEL_SAVE_PATH_SHAPED
            if args.reward_shaping
            else config.MODEL_SAVE_PATH
        )
    else:
        save_path = args.save

    train(
        total_timesteps=args.timesteps,
        save_path=save_path,
        plot_path=args.plot,
        seed=args.seed,
        reward_shaping=args.reward_shaping,
        shaping_beta=args.shaping_beta,
    )


if __name__ == "__main__":
    main()
