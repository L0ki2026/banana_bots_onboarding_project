# Banana Bots RL Navigation Sandbox

Small 2D reinforcement-learning prototype inspired by assistive navigation: a **Gymnasium** environment with partial observations, procedural maps, **PPO** (Stable-Baselines3), held-out evaluation, and optional **Pygame** / notebook visualization.

## Setup

```bash
cd banana_bots
pip install -r banana_bots_rl/requirements.txt
```

Google Colab:

```python
!pip install gymnasium stable-baselines3 pygame matplotlib
```

## Quick start

```bash
# Random baseline on 10 held-out maps
python -m banana_bots_rl.evaluate

# Train PPO (adjust timesteps in config.py or CLI)
python -m banana_bots_rl.train --timesteps 50000
```

After training:

```python
from stable_baselines3 import PPO
from banana_bots_rl.evaluate import compare_agents, print_summary

model = PPO.load("models/ppo_navigation.zip")
results = compare_agents(model, save_path="results/eval.json")
for s in results.values():
    print_summary(s)
```

## Demo visualization (Colab-friendly)

```python
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.visualize import play_episode_pygame, show_frames_in_notebook
from stable_baselines3 import PPO

env = AssistiveNavigationEnv(procedural=True, render_mode="rgb_array")
model = PPO.load("models/ppo_navigation.zip")
frames = play_episode_pygame(env, model=model)
show_frames_in_notebook(frames)
```

## Layout

| File | Role |
|------|------|
| `config.py` | Grid size, rewards, seeds, training defaults |
| `map_generator.py` | Random maps + solvability (BFS) |
| `environment.py` | `AssistiveNavigationEnv` |
| `train.py` | PPO 128×128 MLP, saves model and reward plot |
| `evaluate.py` | Held-out maps, random vs PPO metrics |
| `visualize.py` | RGB frames and optional Pygame |
| `utils.py` | Rolling mean for plots |

Configuration lives in `banana_bots_rl/config.py`. The environment is independent of PPO so other algorithms or baselines can reuse the same API.

See `banana_bots_rl_onboarding_spec.md` for design rationale and stretch goals.
