import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical
from .gae import calc_gae


class QFunc(nn.Module):
    def __init__(self, state_dim):
        """ single action for each state """
        super(QFunc, self).__init__()
        self.affine1 = nn.Linear(state_dim + 1, 32)
        self.affine2 = nn.Linear(32, 64)
        self.affine3 = nn.Linear(64, 1)

    def forward(self, s, a):
        """ s and a are of batch form (b, d) """
        x = torch.cat((s, a), 1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.affine3(x)
        return x


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 32)
        self.affine2 = nn.Linear(32, 64)
        self.affine3 = nn.Linear(64, action_dim)

    def forward(self, s):
        s = F.relu(self.affine1(s))
        s = F.relu(self.affine2(s))
        s = F.softmax(self.affine3(s), dim=-1)
        return s


class PVNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PVNet, self).__init__()

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


class PQNet(nn.Module):
    """ NN that combines the policy and the q network """

    def __init__(self, state_dim, action_dim):
        super(PQNet, self).__init__()

        self.common_layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(32),
            nn.Linear(32, 64),
            nn.ReLU(64)
        )

        self.policy_head = nn.Linear(64, action_dim)
        self.q_head = nn.Linear(64 + 1, 1)

    def forward(self):
        raise NotImplementedError()

    def calc_q(self, s, a):
        s = self.common_layer(s)
        x = torch.cat((s, a), 1)
        q_values = self.q_head(x)
        return q_values

    def calc_policy(self, s):
        s = self.common_layer(s)
        action_probs = F.softmax(self.policy_head(s), dim=-1)
        return action_probs


class TrajCVPolicy:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.gamma = gamma
        self.q_update_epochs = 10

        self.epsilon_greedy_threshold = 15
        self.epsilon_anneal = 0.01

        self.net = PVNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.saved_logprobs = []
        self.saved_rewards = []
        self.saved_state_values = []

        self.action_dim = action_dim

    def select_action(self, s):
        """ take in the raw representation of state from the env and return an action """
        s = torch.from_numpy(s).float()
        action_prob, state_value = self.net.forward(s)

        m = Categorical(action_prob)

        # epsilon greedy
        # if np.random.randint(100) < self.epsilon_greedy_threshold:
        #     a = torch.IntTensor([np.random.randint(self.action_dim)]).squeeze(-1)
        # else:
        #     a = m.sample()
        a = m.sample()

        # save the v
        self.saved_state_values.append(state_value)
        # save the log probability of the action
        self.saved_logprobs.append(m.log_prob(a))

        return a.item()

    def save_reward(self, r):
        self.saved_rewards.append(r)

    def save_state_action(self, s, a, done):
        if done:
            # to save the v value for the last state
            # s = torch.from_numpy(s).float()
            # _, state_value = self.net.forward(s)
            # self.saved_state_values.append(state_value)
            self.saved_state_values.append(torch.FloatTensor([0]))

    def finish_episode(self):
        """ called at the end of each trajectory """
        R = 0
        returns = []
        policy_losses = []
        v_losses = []

        for r in self.saved_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        # find advantage
        advantages = calc_gae(torch.FloatTensor(self.saved_rewards), torch.cat(self.saved_state_values))
        # advantages = (advantages - advantages.mean()) / (advantages.std() + np.finfo(np.float32).eps.item())

        for i in reversed(range(1, len(advantages))):
            advantages[i - 1] += advantages[i]

        signals = returns - advantages

        for r, signal, v, log_prob in zip(returns, signals, self.saved_state_values[:-1], self.saved_logprobs):
            policy_losses.append(-signal.detach() * log_prob)
            v_losses.append(F.smooth_l1_loss(v.squeeze(-1), r))

        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(v_losses).sum()
        loss.backward()

        self.optimizer.step()

        del self.saved_logprobs[:]
        del self.saved_rewards[:]
        del self.saved_state_values[:]

        self.epsilon_greedy_threshold = max(0, self.epsilon_greedy_threshold - self.epsilon_anneal)

