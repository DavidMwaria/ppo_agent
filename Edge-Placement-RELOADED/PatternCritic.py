import torch
import torch.nn as nn


# obs_dim: 51 action_dim: the number of patterns in each m-n config

class PatternCritic22(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim22):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim22
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim22)

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


class PatternCritic32(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim32
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim32)

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


class PatternCritic42(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim42):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim42
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim42)

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


class PatternCritic23(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim23):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim23
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim23)

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


class PatternCritic33(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim33):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim33
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim33)

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


class PatternCritic43(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim43):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim43
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim43)

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


class PatternCritic24(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim24):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim24
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim24)

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


class PatternCritic34(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim34):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim34
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim34)

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


class PatternCritic44(nn.Module):
    def __init__(self, obs_dim, pattern_action_dim44):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = pattern_action_dim44
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(obs_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, pattern_action_dim44)

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
