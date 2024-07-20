import torch
import torch.nn as nn


# obs_dim: 51 action_dim: 4 : 3 path + 1 rej

class PathCritic(nn.Module):
    def __init__(self, obs_dim, path_action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = path_action_dim
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, path_action_dim)

    def forward(self, batch_obs):
        out = self.fc1(batch_obs)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.fc4(out)
        out = torch.relu(out)
        out = self.fc5(out)
        out = torch.relu(out)
        return out