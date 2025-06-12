# Snake-AI Tutorial

This guide walks you through setting up the project, playing the game and training the AI agent.

## 1. Installation

1. Install **Python 3.8** or later.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Playing the Game

Run the following command to start the classic Snake game:

```bash
python game.py
```

Use the arrow keys to control the snake. Game options such as obstacle count and screen size are stored in `config.json`.

## 3. Training an Agent

To train the DQN agent use the training script:

```bash
python training.py --config config.json [--render]
```

The optional `--render` flag displays the game window while training. A new weights file `dqn_snake_model_<name>.pth` will be written after training completes.

You can also pass a directory path to `--config`; in that case all JSON files in the directory are used for training runs.

## 4. Running a Trained Agent

Load a saved model and watch it play by running:

```bash
python bench.py
```

Make sure the weight file (e.g. `dqn_snake_model.pth`) is present in the current directory.

## 5. Custom Configuration

All tunable parameters are stored in `config.json`. You may create copies of this file with different settings and specify them via `--config` for both `game.py` and `training.py`.

That's it! You are ready to experiment with and extend the Snake-AI project.
