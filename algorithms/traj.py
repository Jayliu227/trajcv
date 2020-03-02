import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical


class QFunc(nn.Module):
    def __init__(self, state_dim):
        """ single action for each state """
        super(QFunc, self).__init__()
        self.affine1 = nn.Linear(state_dim + 1, 64)
        self.affine2 = nn.Linear(64, 128)
        self.affine3 = nn.Linear(128, 1)

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
        self.affine1 = nn.Linear(state_dim, 64)
        self.affine2 = nn.Linear(64, 128)
        self.affine3 = nn.Linear(128, action_dim)

    def forward(self, s):
        s = F.relu(self.affine1(s))
        s = F.relu(self.affine2(s))
        s = F.softmax(self.affine3(s), dim=-1)
        return s


class TrajCVPolicy:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.gamma = gamma
        self.q_update_epochs = 100

        self.policy = Policy(state_dim, action_dim)
        self.Q = QFunc(state_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr * 3)

        self.saved_logprobs = []
        self.saved_qvalues = []
        self.saved_expectedqs = []
        self.saved_rewards = []
        self.saved_actions = []
        self.saved_states = []

        self.action_dim = action_dim

    def select_action(self, s):
        """ take in the raw representation of state from the env and return an action """
        s = torch.from_numpy(s).float()
        action_prob = self.policy.forward(s)

        m = Categorical(action_prob)
        a = m.sample()

        all_states = torch.stack([s] * self.action_dim)
        all_actions = torch.FloatTensor([[i] for i in range(self.action_dim)])

        all_qvalues = self.Q.forward(all_states, all_actions)
        expected_qvalue = all_qvalues.mean()

        # save the expected qvalue
        self.saved_expectedqs.append(expected_qvalue)
        # save the qvalue of the action we take
        self.saved_qvalues.append(all_qvalues[a.item()].squeeze(-1))
        # save the log probability of the action
        self.saved_logprobs.append(m.log_prob(a))

        return a.item()

    def save_reward(self, r):
        self.saved_rewards.append(r)

    def save_state_action(self, s, a):
        self.saved_actions.append(torch.FloatTensor([a]))
        self.saved_states.append(torch.FloatTensor(s))

    def finish_episode(self):
        """ called at the end of each trajectory """
        R = 0
        returns = []
        policy_losses = []
        q_losses = []

        for r in self.saved_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        # find advantage
        advantages = torch.FloatTensor(self.saved_qvalues) - torch.FloatTensor(self.saved_expectedqs)

        for i in reversed(range(1, len(self.saved_qvalues))):
            advantages[i - 1] += advantages[i]

        for r, q, advantage, log_prob in zip(returns, self.saved_qvalues, advantages, self.saved_logprobs):
            g = r - advantage.detach()
            policy_losses.append(-g * log_prob)
            q_losses.append(F.smooth_l1_loss(q, r))

        self.policy_optimizer.zero_grad()
        self.Q_optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(q_losses).sum()
        loss.backward()

        self.policy_optimizer.step()
        self.Q_optimizer.step()

        # update Q several more times
        prev_actions = torch.stack(self.saved_actions)
        prev_states = torch.stack(self.saved_states)
        for i in range(self.q_update_epochs):
            q_values = self.Q.forward(prev_states, prev_actions).squeeze(-1)

            self.Q_optimizer.zero_grad()
            loss = F.mse_loss(q_values, returns)
            loss.mean().backward()
            self.Q_optimizer.step()

        del self.saved_logprobs[:]
        del self.saved_qvalues[:]
        del self.saved_expectedqs[:]
        del self.saved_rewards[:]
        del self.saved_actions[:]
        del self.saved_states[:]
