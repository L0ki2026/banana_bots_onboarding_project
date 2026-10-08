"""Manual smoke test for the environment (no RL). Run: PYTHONPATH=. python -m banana_bots_rl.manual_test_env"""

from __future__ import annotations

import numpy as np

from banana_bots_rl import config
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.map_generator import generate_map


def main() -> None:
    try:
        from gymnasium.utils.env_checker import check_env

        check_env(AssistiveNavigationEnv(procedural=True), skip_render_check=True)
        print("Gymnasium env checker: OK")
    except Exception as exc:
        print(f"Env checker: {exc}")

    env = AssistiveNavigationEnv(procedural=True)
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)

    layout = generate_map(rng=np.random.default_rng(0))
    env.reset(options={"layout": layout})
    total = 0.0
    for _ in range(config.MAX_STEPS):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        total += reward
        if term:
            print("Reached goal; reward snippet:", total)
            break
        if trunc:
            print("Timeout; collisions:", info["collisions_total"])
            break

    frame = env._render_frame()
    print("Render frame shape:", frame.shape)


if __name__ == "__main__":
    main()
