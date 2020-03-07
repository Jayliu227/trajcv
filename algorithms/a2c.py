import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(state_dim, 64)
        self.affine2 = nn.Linear(64, 128)

        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_probs, state_values


class A2C:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.gamma = gamma
        self.q_update_epochs = 100

        self.policy = Policy(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.saved_actions = []
        self.saved_rewards = []
        # self.saved_states = []

    def select_action(self, s):
        s = torch.from_numpy(s).float()
        action_prob, state_value = self.policy.forward(s)

        m = Categorical(action_prob)
        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def save_reward(self, r):
        self.saved_rewards.append(r)

    def save_state_action(self, s, a, done):
        # self.saved_states.append(torch.FloatTensor(s))
        pass

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

        for (log_prob, state_value), R in zip(self.saved_actions, returns):
            advantage = R - state_value.item()

            policy_losses.append(-log_prob * advantage)

            value_losses.append(F.smooth_l1_loss(state_value, torch.tensor([R])))

        self.policy_optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()

        self.policy_optimizer.step()

        # prev_states = torch.stack(self.saved_states)
        # for i in range(self.q_update_epochs):
        #     _, state_values = self.policy.forward(prev_states)
        #
        #     self.policy_optimizer.zero_grad()
        #     loss = F.mse_loss(state_values.squeeze(-1), returns)
        #     loss.mean().backward()
        #     self.policy_optimizer.step()

        del self.saved_actions[:]
        del self.saved_rewards[:]
        # del self.saved_states[:]
