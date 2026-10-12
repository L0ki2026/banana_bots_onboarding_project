"""Central configuration for the RL navigation sandbox."""

# Grid
GRID_SIZE = 12

# Partial visibility: radius in cells for local observation window
VIEW_DISTANCE = 5

# Episode
MAX_STEPS = 200

# Rewards
GOAL_REWARD = 100.0
COLLISION_PENALTY = -20.0
GRASS_PENALTY = -1.0
STEP_PENALTY = -0.1

# Map generation
OBSTACLE_DENSITY = 0.12
GRASS_DENSITY = 0.18
MIN_PATH_LENGTH = 3  # minimum Manhattan distance start–goal when sampling

# Reproducibility
RANDOM_SEED = 42
EVAL_SEED = 12345
NUM_EVAL_MAPS = 100

# Training (defaults; override in Colab/notebook as needed)
TOTAL_TIMESTEPS = 200_000
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_GAMMA = 0.99
MODEL_SAVE_PATH = "models/ppo_navigation.zip"
MODEL_SAVE_PATH_SHAPED = "models/ppo_navigation_shaped.zip"

# Potential-based reward shaping (optional)
REWARD_SHAPING_ENABLED = False
REWARD_SHAPING_BETA = 1.0

# Visualization
CELL_PIXEL_SIZE = 40
FPS = 8

# Cell type indices (for grid and observations)
EMPTY = 0
GRASS = 1
OBSTACLE = 2
GOAL = 3
# Agent position is tracked separately; goal cell type remains GOAL on grid until reached
