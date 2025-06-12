# Snake-AI

This project contains a small reinforcement learning playground built around the
classic Snake game.  Besides a playable version of the game it includes a
`snake_env.py` environment and a simple Deep Q-Network (DQN) agent used for
experiments.

For a step-by-step introduction see [TUTORIAL.md](TUTORIAL.md).

## Features

- Play a classic snake game with animated UI (`game.py`).
- Gym-like environment (`snake_env.py`) for training RL agents.
- DQN training script (`training.py`).
- Example agent weights (`dqn_snake_model.pth`).
- Script to run the trained model (`bench.py`).
- Customisable environment via `config.json`.

## Installation

1. Install Python **3.8** or later.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

The requirements file installs both `pygame` for rendering and `torch` for the DQN implementation.

## Running the Game

To play the Snake game with keyboard controls run:

```bash
python game.py
```

The configuration in `config.json` controls various game options such as obstacle count, movement speed and window size.

## Training the AI

The file `training.py` trains a DQN agent using `SnakeEnv`. The script reads the same configuration file as the game. Training can be started with:

```bash
python training.py --config config.json [--render]
```
The optional `--render` flag shows the environment during training. When enabled, the same graphical interface used by `bench.py` and the regular game is displayed so you can watch the learning progress.

A path to a directory may also be supplied, in which case all JSON files in that directory are used for training runs.

After training finishes a model file `dqn_snake_model_<name>.pth` is written containing the learned weights.

## Running the Trained Agent

A small script `bench.py` loads a previously saved weight file and lets the agent play using the standard game interface:

```bash
python bench.py
```

This launches the regular game window but actions are chosen by the trained model
instead of keyboard input.

Ensure that `dqn_snake_model.pth` (or your own saved weights) is present in the working directory.

## Configuration

`config.json` holds all tunable parameters. Important options include:

- `WINDOW_WIDTH`, `WINDOW_HEIGHT`, `CELL_SIZE` – dimensions of the game grid.
- `USE_STATIC_OBSTACLES`, `NUM_STATIC_OBSTACLE_CLUSTERS`, `CLUSTER_MAX_SIZE` – control randomly generated obstacles.
- `USE_MOVING_OBSTACLES`, `INITIAL_MOVING_OBSTACLES_COUNT` – add moving hazards.
- `USE_LIVES`, `INITIAL_LIVES` – optional life system in the manual game.
- `BASE_FPS` – baseline game speed.

You may duplicate this file under different names to experiment with various settings and then pass the file path using `--config` when training or playing.

## Repository Structure

```
bench.py      # Run a trained model
config.json   # Default configuration
game.py       # Playable Snake game
snake_env.py  # Gym-like environment used for training
training.py   # DQN training script
textures/     # Sprite images
TUTORIAL.md   # Tutorial for more information
```

## License

This project is provided as-is for educational purposes. Textures originate from open-source or freely available assets.

