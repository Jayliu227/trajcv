import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from torch.distributions import Categorical
from math import ceil

from .models import PVNet
from .replay_buffer import RE

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2C:
    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma=0.99,
                 lr=1e-3,
                 v_update_epochs=30.0,
                 v_update_anneal=0.00,
                 epsilon_greedy_threshold=10.0,
                 epsilon_anneal=0.01,
                 re_sample_batch_size=100
                 ):
        self.gamma = gamma

        self.v_update_epochs = v_update_epochs
        self.v_update_anneal = v_update_anneal

        self.epsilon_greedy_threshold = epsilon_greedy_threshold
        self.epsilon_anneal = epsilon_anneal

        self.action_dim = action_dim

        self.state_RE = RE(re_sample_batch_size)

        self.net = PVNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.saved_actions = []
        self.saved_rewards = []
        self.saved_states = []

    def select_action(self, s):
        s = torch.from_numpy(s).float()
        action_prob, state_value = self.net.forward(s)

        m = Categorical(action_prob)

        if np.random.randint(100) < ceil(self.epsilon_greedy_threshold):
            a = torch.IntTensor([np.random.randint(self.action_dim)]).squeeze(-1)
        else:
            a = m.sample()
        # a = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(a), state_value))

        return a.item()

    def save_reward(self, r):
        self.saved_rewards.append(r)

    def save_state_action(self, s, a, done):
        s = torch.from_numpy(s).float()
        self.saved_states.append(torch.FloatTensor(s))

    def finish_episode(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.saved_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        for (s, v) in zip(self.saved_states, returns):
            self.state_RE.add(s, v)

        for (log_prob, state_value), R in zip(self.saved_actions, returns):
            advantage = R - state_value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(state_value, torch.tensor([R])))

        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()

        self.optimizer.step()

        for i in range(ceil(self.v_update_epochs)):
            old_states, old_values = self.state_RE.sample()
            old_states, old_values = torch.stack(old_states), torch.stack(old_values)

            state_values = self.net.calc_v(old_states)
            loss = F.mse_loss(old_values, state_values.squeeze(-1))
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        del self.saved_actions[:]
        del self.saved_rewards[:]
        del self.saved_states[:]

        self.state_RE.delete_random()
        self.epsilon_greedy_threshold = max(0, self.epsilon_greedy_threshold - self.epsilon_anneal)
        self.v_update_epochs = max(0, self.v_update_epochs - self.v_update_anneal)

