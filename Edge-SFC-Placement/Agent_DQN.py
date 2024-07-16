# implementation of Agent using DQN

from torch.nn.modules import loss
from torch.nn.modules.loss import L1Loss
import collections
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from PathCritic import *


class RepMem(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch \
            = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        return torch.from_numpy(np.array(obs_batch).astype('float32')), \
            torch.from_numpy(np.array(action_batch).astype('int64')), \
            torch.from_numpy(np.array(reward_batch).astype('float32')).view(-1, 1), \
            torch.from_numpy(np.array(next_obs_batch).astype('float32')), \
            torch.from_numpy(np.array(done_batch).astype('float32'))

    def __len__(self):
        return len(self.buffer)


class Agent():
    def __init__(self,
                 critic,
                 obs_dim,
                 action_dim,
                 lr,
                 gamma,
                 alpha,
                 update_target_steps=200):
        self.e_greed = 0.1
        self.e_greed_decrement = 1e-6
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.global_step = 0
        self.update_target_steps = update_target_steps
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)
        self.criteria_critic = nn.MSELoss()

    def path_predict(self, obs):  # select best action
        # print(type(obs),'len=',len(obs))
        obs = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        # print(type(obs),'len=',len(obs))
        with torch.no_grad():
            return self.critic(obs).argmax(dim=1).item()

    def sample(self, obs):
        sample = np.random.rand()
        if sample < self.e_greed:
            path_act = np.random.randint(self.action_dim)
        else:
            path_act = self.path_predict(obs)

        return path_act

    def sync_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def learn(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal):
        # synchronize parameters of model and target_model every 200 training steps
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1

        # print('batch action: ', batch_action)
        # print('num classes: ', self.action_dim)
        # train critic 
        pred_value = (self.critic(batch_obs) *
                      F.one_hot(batch_action, num_classes=self.action_dim)) \
            .sum(dim=1).view(-1, 1)  # get predicted Q-value

        with torch.no_grad():
            # argmax action of critic(s_{t+1})
            max_action = self.critic(batch_next_obs).argmax(dim=1)
            one_hot_max_action = F.one_hot(max_action, num_classes=self.action_dim)
            target_q = (self.target_critic(batch_next_obs) * one_hot_max_action).sum(dim=1, keepdim=True)
            target_value = batch_reward + (1 - batch_terminal.view(-1, 1)) * target_q * self.gamma
            target_value = (target_value - pred_value) * self.alpha + pred_value
        loss_critic = self.criteria_critic(pred_value, target_value)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

    def save(self, path):
        torch.save(self.critic.state_dict(), path)

    def load(self, path):
        self.critic.load_state_dict(torch.load(path))
