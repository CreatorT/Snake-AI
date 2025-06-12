import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import glob
import argparse
from collections import deque
from snake_env import SnakeEnv  # gym-like snake environment

# Select device: CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
STATE_SIZE = 17      
ACTION_SIZE = 4      # UP, DOWN, LEFT, RIGHT
LR = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
NUM_EPISODES = 10000
UPDATE_TARGET_EVERY = 10

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START

        # Initialize networks and move them to the selected device
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train(config_path: str = "config.json", render: bool = False):
    """Train a DQN agent using the given environment configuration.

    Parameters
    ----------
    config_path : str
        Path to a configuration JSON file.
    render : bool, optional
        Whether to render the environment during training. Rendering slows
        down training but can be useful for debugging.
    """

    env = SnakeEnv(render_mode=render, config_path=config_path)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target()

        print(
            f"Episode {episode} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}"
        )

    # Save the trained model (only the state dict is saved)
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    model_path = f"dqn_snake_model_{cfg_name}.pth"
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Snake RL agent")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to a config file or directory containing config files",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training",
    )
    args = parser.parse_args()

    config_input = args.config
    if os.path.isdir(config_input):
        config_paths = sorted(glob.glob(os.path.join(config_input, "*.json")))
    else:
        config_paths = [config_input]

    for cfg in config_paths:
        print(f"\n=== Training with {cfg} ===")
        train(cfg, render=args.render)
