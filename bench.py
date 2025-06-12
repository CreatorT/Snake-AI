import torch
from training import QNetwork
from game import game_loop

# Hyperparameters
STATE_SIZE = 17
ACTION_SIZE = 4

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a new model instance and load the trained weights
model = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
model.load_state_dict(torch.load("dqn_snake_model.pth", map_location=device))
model.eval()

game_loop(model=model, device=device)
