# Banana Bots Onboarding Project: RL Navigation Evaluation Sandbox

## 1. Project Goal

Build a small **2D reinforcement-learning navigation sandbox** inspired
by Banana Bots' assistive navigation glasses project.

The real Banana Bots system uses a computer-vision model to interpret
the environment. This onboarding prototype **does not recreate or
replace their vision model**. Instead, it creates a controlled simulated
environment and a placeholder navigation agent to demonstrate how
reinforcement learning can be used to test navigation behavior across
varied and previously unseen environments.

The key question is:

> Can an RL navigation policy trained across many randomized
> environments learn to navigate safely and generalize to layouts it has
> never encountered before?

This is a prototype and should stay intentionally simple, modular, and
easy to explain.

------------------------------------------------------------------------

## 2. Technology Stack

-   **Language:** Python
-   **Runtime:** Google Colab compatible
-   **Environment API:** Gymnasium
-   **RL library:** Stable-Baselines3
-   **RL algorithm:** PPO (Proximal Policy Optimization)
-   **Policy network:** MLP
-   **Visualization:** Pygame
-   **Metrics/plots:** Matplotlib
-   **Optional numerical utilities:** NumPy

Important terminology:

-   **Gymnasium** defines the simulated environment.
-   **Stable-Baselines3** provides the RL implementation.
-   **PPO** is the reinforcement-learning algorithm that trains the
    policy.
-   The **MLP** is the neural-network policy trained by PPO.
-   **Pygame** renders the navigation environment.
-   **Matplotlib** plots training/evaluation metrics.

------------------------------------------------------------------------

## 3. High-Level Architecture

``` text
Random 2D Environment
        |
        v
Local Observation / Simulated Perception
        |
        v
MLP Policy Network
(128 -> 128, ReLU)
        |
        v
Navigation Action
        |
        v
Gymnasium Environment
        |
        v
Reward + Next Observation
        |
        +--------------------+
        |                    |
        v                    |
       PPO ------------------+
   updates policy
```

The simulation acts as a simplified stand-in for the real system:

``` text
REAL SYSTEM:
Camera -> Computer Vision Model -> Environmental Information -> Navigation Logic

THIS PROTOTYPE:
2D Simulator -> Local Observation -> PPO-trained Navigation Policy
```

The code should be modular enough that a future version could replace
the simulated observation with output from an actual computer-vision
model.

------------------------------------------------------------------------

## 4. 2D Environment

Create a custom Gymnasium environment.

Suggested class:

``` python
class AssistiveNavigationEnv(gym.Env):
    ...
```

### Initial grid size

Start with:

``` text
12 x 12
```

Make grid size configurable so it can later be changed.

### Cell types

Keep V1 simple:

``` text
EMPTY / PATH
GRASS
OBSTACLE
GOAL
AGENT
```

Obstacles can conceptually represent real outdoor objects such as:

-   poles
-   trash cans
-   benches
-   walls/barriers

For V1, they can all use one generic `OBSTACLE` class internally.

------------------------------------------------------------------------

## 5. Partial Visibility / Line of Sight

The agent must **not receive the entire map**.

This is important because assistive glasses would make navigation
decisions based on what is currently visible rather than receiving a
perfect top-down map.

Use a local observation corresponding to approximately **5 grid cells of
line of sight**.

For the initial implementation, make this easy to modify. A practical
representation could be a local window centered/oriented around the
agent or a forward-facing local sensor representation.

The agent should know enough to make local navigation decisions, but it
should not be given the complete layout.

Also include a simple representation of the direction toward the
destination, such as:

``` text
delta_x_to_goal
delta_y_to_goal
```

or a normalized goal-direction vector.

Do **not** give the exact full map to the policy.

------------------------------------------------------------------------

## 6. Action Space

Use a discrete action space.

V1 actions:

``` text
0 = forward
1 = backward
2 = left
3 = right
```

The environment should maintain the agent's orientation if `forward`,
`backward`, `left`, and `right` are interpreted relative to where it is
facing.

If that makes the first implementation unnecessarily complex, use four
cardinal movements initially:

``` text
up
down
left
right
```

but structure the code so orientation-based actions can be added easily.

Preferred final V1 behavior: **forward, backward, left, right relative
to orientation**.

------------------------------------------------------------------------

## 7. Policy Network

Use Stable-Baselines3 PPO with an MLP policy.

Desired hidden architecture:

``` text
Input
  |
128 units + ReLU
  |
128 units + ReLU
  |
Policy output
```

Use Stable-Baselines3 policy configuration rather than manually
implementing PPO.

Conceptually:

``` python
policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=torch.nn.ReLU
)
```

Then:

``` python
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    ...
)
```

Do not implement PPO from scratch.

------------------------------------------------------------------------

## 8. Reward Function

Keep the initial reward system intentionally simple.

### Reach goal

``` text
+100
```

### Collision with obstacle

``` text
-20
```

The collision should prevent the agent from moving through the obstacle.

### Walking on grass

Use a small negative reward:

``` text
-1
```

This represents grass being less desirable than a clear path but not
necessarily unsafe.

### Normal movement

Use a very small step penalty so the agent is encouraged to find
efficient routes:

``` text
-0.1 per step
```

### Initial reward table

  Event                   Reward
  --------------------- --------
  Reach destination         +100
  Collision                  -20
  Enter/walk on grass         -1
  Normal step               -0.1

Avoid complicated reward shaping in V1.

All reward values should be stored as configurable constants.

------------------------------------------------------------------------

## 9. Episode Termination

An episode ends when:

1.  The agent reaches the goal, OR
2.  A maximum number of steps is reached.

Example:

``` text
MAX_STEPS = 200
```

Make this configurable.

A collision does not necessarily need to terminate the episode.
Initially, give the collision penalty and leave the agent in its
previous valid position.

------------------------------------------------------------------------

## 10. Random Environment Generation

Do not create a conventional fixed supervised-learning dataset.

Instead, generate layouts procedurally.

For each training episode:

1.  Create a new grid.
2.  Randomly place the start position.
3.  Randomly place the destination.
4.  Add obstacles.
5.  Add grass cells/regions.
6.  Ensure the map is solvable.
7.  Run the episode.

The map generator should guarantee that at least one valid path exists
between the start and goal.

Use a random seed when appropriate so experiments can be reproduced.

------------------------------------------------------------------------

## 11. Training vs. Generalization Evaluation

The goal is not simply to measure performance on environments the policy
repeatedly saw.

### Training

Train PPO across many randomized layouts generated on the fly.

Example target:

``` text
hundreds or thousands of randomized episodes
```

Do not hard-code an exact number until basic training behavior is
working.

### Held-out evaluation

Generate a separate fixed collection of random maps using a different
seed.

These maps must **not be used for training**.

For example:

``` text
Training: procedurally generated maps
Evaluation: 100 fixed unseen maps
```

Evaluate the trained policy on the unseen maps.

This is the core generalization experiment.

------------------------------------------------------------------------

## 12. Evaluation Metrics

Track at least:

### Success rate

``` text
successful episodes / total evaluation episodes
```

### Collision rate

Track either:

``` text
average collisions per episode
```

and/or

``` text
percentage of episodes containing a collision
```

### Average steps to goal

For successful episodes:

``` text
mean number of steps required to reach destination
```

### Average episode reward

Track cumulative reward per episode.

Optional later metrics:

-   percentage of movement spent on grass
-   path efficiency compared with shortest possible path
-   failure rate
-   timeout rate

------------------------------------------------------------------------

## 13. Matplotlib Plots

Use Matplotlib for training/evaluation graphs.

At minimum produce:

### Training reward

``` text
X-axis: training timesteps / episodes
Y-axis: average episode reward
```

Prefer a rolling average rather than plotting only noisy raw rewards.

Also useful:

``` text
Success Rate vs Training Progress
Collision Rate vs Training Progress
Average Steps vs Training Progress
```

The first implementation only needs the reward plot plus final
evaluation metrics.

------------------------------------------------------------------------

## 14. Pygame Visualization

Use Pygame to visually render selected episodes.

Suggested visual representation:

``` text
Agent      -> directional icon/triangle
Goal       -> target marker
Obstacle   -> block
Grass      -> textured/marked cell
Path       -> empty/light cell
```

The visualization should show the agent moving step by step.

Do **not** render every training episode because that will slow
training.

Instead:

``` text
Training -> rendering disabled
Demo/evaluation -> rendering enabled
```

Provide a way to run one evaluation map visually after training.

------------------------------------------------------------------------

## 15. Google Colab Compatibility

The entire project must run in Google Colab.

Include an installation cell or setup section equivalent to:

``` bash
pip install gymnasium stable-baselines3 pygame
```

Matplotlib and NumPy may already be installed but can be included if
necessary.

Because Colab does not provide a normal desktop display, Pygame
rendering must be designed with Colab limitations in mind.

Preferred approach:

-   Environment training works completely headlessly.
-   Rendering is optional.
-   Render frames to arrays/images when running in Colab.
-   Display captured frames inside the notebook rather than requiring a
    native Pygame window.

Do not make training depend on GUI availability.

------------------------------------------------------------------------

## 16. Suggested Project Structure

Create clean modular files:

``` text
banana_bots_rl/
|
|-- environment.py
|-- map_generator.py
|-- train.py
|-- evaluate.py
|-- visualize.py
|-- config.py
|-- utils.py
|-- requirements.txt
|-- README.md
```

### environment.py

Contains:

``` text
AssistiveNavigationEnv
observation space
action space
step()
reset()
reward logic
termination logic
```

### map_generator.py

Contains:

``` text
random grid generation
obstacle generation
grass generation
start/goal placement
solvability check
```

### train.py

Contains:

``` text
environment creation
PPO configuration
128x128 ReLU MLP configuration
training
model saving
training metrics
```

### evaluate.py

Contains:

``` text
held-out map generation/loading
success-rate calculation
collision statistics
average steps
average reward
```

### visualize.py

Contains:

``` text
Pygame rendering
Colab-compatible frame rendering
demo episode
```

### config.py

Contains configurable parameters:

``` text
GRID_SIZE
VIEW_DISTANCE
MAX_STEPS
GOAL_REWARD
COLLISION_PENALTY
GRASS_PENALTY
STEP_PENALTY
OBSTACLE_DENSITY
GRASS_DENSITY
RANDOM_SEED
```

------------------------------------------------------------------------

## 17. Development Order

Build the project incrementally.

### Phase 1 - Environment

Implement the Gymnasium environment without RL.

Verify manually that:

-   maps generate correctly
-   the agent moves correctly
-   obstacles block movement
-   grass is detected
-   rewards are correct
-   goal terminates an episode
-   maximum steps terminate an episode

### Phase 2 - Random Maps

Implement procedural map generation and guarantee solvability.

### Phase 3 - PPO

Connect the environment to Stable-Baselines3 PPO.

Use:

``` text
MlpPolicy
128 -> 128
ReLU
```

Train a basic model.

### Phase 4 - Evaluation

Create held-out maps and calculate:

``` text
success rate
collision rate
average steps
average reward
```

### Phase 5 - Visualization

Render representative successful and failed navigation episodes.

### Phase 6 - Plots

Plot training reward and evaluation metrics.

------------------------------------------------------------------------

## 18. Important Design Principle

Keep the environment independent from PPO.

The Gymnasium environment should expose the standard interface:

``` python
observation, reward, terminated, truncated, info = env.step(action)
```

The environment should not contain PPO-specific learning logic.

This allows the same environment to later evaluate:

-   PPO
-   DQN
-   A2C
-   heuristic navigation
-   random policy
-   another trained navigation model

This makes the sandbox genuinely useful as an **evaluation framework**,
rather than just a PPO demo.

------------------------------------------------------------------------

## 19. Baseline Agent

Before training PPO, implement a simple baseline.

At minimum include:

``` text
Random Agent
```

It selects a random legal action.

Evaluate it on the same held-out maps.

Then compare:

``` text
Random Agent vs PPO Agent
```

Metrics:

``` text
Success Rate
Collision Rate
Average Steps
Average Reward
```

This gives the experiment a meaningful comparison and demonstrates
whether RL actually improved navigation.

------------------------------------------------------------------------

## 20. Connection to Banana Bots

The project should be explained as a **small proof of concept**, not as
a solution to Banana Bots' entire navigation problem.

The conceptual connection is:

``` text
Banana Bots:
Camera
  ->
Computer Vision Model
  ->
Perception / Environmental Representation
  ->
Navigation Decision
  ->
Instruction to User
```

This prototype focuses on:

``` text
Simulated Environmental Observation
  ->
RL Navigation Policy
  ->
Navigation Action
```

The simulator is therefore a controlled sandbox for studying navigation
behavior.

A future integration could replace the simulated observation with
environmental information generated by Banana Bots' actual vision model.

------------------------------------------------------------------------

## 21. What V1 Is NOT Trying to Do

Do not add these unless the basic system is already complete:

-   real camera input
-   object-detection training
-   vision-transformer training
-   LLM integration
-   3D simulation
-   Unity
-   Unreal Engine
-   ROS
-   real UCSC map reconstruction
-   realistic pedestrian simulation
-   RNN/LSTM memory
-   artificial perception confidence/noise
-   complex reward shaping

These are possible extensions, not requirements.

------------------------------------------------------------------------

## 22. Stretch Goals

Only after V1 works:

### Memory

Compare:

``` text
MLP PPO
vs.
recurrent policy / LSTM-based navigation
```

This could test whether memory helps when the environment is only
partially visible.

### Indoor vs outdoor layouts

Create two map-generation distributions:

``` text
Indoor:
corridors
walls
door-like openings
tight obstacle spacing

Outdoor:
open areas
grass
isolated poles
benches
wider paths
```

Train on one distribution and evaluate on the other.

This would connect especially well to the original Banana Bots issue of
a model that worked in UCSC hallways but struggled around the East
Field.

### Different visibility distances

Compare:

``` text
3-cell vision
5-cell vision
7-cell vision
```

### Different obstacle densities

Evaluate whether performance degrades as navigation becomes harder.

------------------------------------------------------------------------

## 23. Final Demo

The final onboarding demonstration should ideally show:

1.  A randomly generated 2D environment.
2.  The agent's limited local observation.
3.  A random baseline attempting navigation.
4.  The PPO-trained agent attempting the same type of navigation.
5.  A Pygame/Colab visualization of the trained policy.
6.  A Matplotlib training-reward curve.
7.  Evaluation on held-out unseen maps.
8.  Final metrics such as success rate, collisions, average steps, and
    reward.

The presentation should emphasize:

> This is an RL-based navigation evaluation sandbox inspired by the
> assistive glasses project. It tests whether a navigation policy
> trained in randomized simulated environments can generalize to unseen
> layouts, while keeping the simulation modular enough that simulated
> observations could later be replaced with outputs from a real
> computer-vision system.

------------------------------------------------------------------------

## 24. Instructions for Cursor

Build this project **incrementally**. Do not generate an unnecessarily
large framework in one step.

Priorities:

1.  Correctness
2.  Simplicity
3.  Google Colab compatibility
4.  Modular code
5.  Easy-to-understand implementation
6.  Reproducible evaluation

Use comments to explain important RL/environment decisions, but avoid
excessive comments explaining trivial Python syntax.

Before adding PPO, first verify the custom Gymnasium environment using
manual/random actions.

Before training, use Gymnasium's environment checker where possible.

Keep configuration values centralized.

Save the trained PPO model and evaluation results.

Do not introduce additional ML frameworks or complex dependencies unless
necessary.

The initial goal is a clean, working V1 that can later be extended.
