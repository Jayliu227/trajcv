import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from math import ceil
from torch.distributions import Categorical
from .gae import calc_gae
from .models import PQNet
from .replay_buffer import RE


class TrajCVPolicy:
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

        self.net = PQNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.state_RE = RE(re_sample_batch_size)

        self.saved_states = []
        self.saved_logprobs = []
        self.saved_rewards = []
        self.saved_state_values = []
        self.saved_qvalues = []
        self.saved_actions = []

        self.action_dim = action_dim

    def select_action(self, s):
        """ take in the raw representation of state from the env and return an action """
        s = torch.from_numpy(s).float()
        action_prob = self.net.calc_policy(s)

        m = Categorical(action_prob)

        # epsilon greedy
        if np.random.randint(100) < ceil(self.epsilon_greedy_threshold):
            a = torch.IntTensor([np.random.randint(self.action_dim)]).squeeze(-1)
        else:
            a = m.sample()
        # a = m.sample()

        all_states = torch.stack([s] * self.action_dim)
        all_actions = torch.FloatTensor([[i] for i in range(self.action_dim)])

        all_qvalues = self.net.calc_q(all_states, all_actions).squeeze(-1)
        expected_qvalue = (all_qvalues * action_prob).sum()

        self.saved_state_values.append(expected_qvalue)
        self.saved_qvalues.append(all_qvalues[a.item()])
        self.saved_logprobs.append(m.log_prob(a))

        return a.item()

    def save_reward(self, r):
        self.saved_rewards.append(r)

    def save_state_action(self, s, a, done):
        s = torch.from_numpy(s).float()
        self.saved_actions.append(torch.FloatTensor([a]))
        self.saved_states.append(s)

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

        # add state and value pairs into the replay buffer
        for (s, a, q) in zip(self.saved_states, self.saved_actions, returns):
            self.state_RE.add((s, a), q)

        # find advantage
        advantages = torch.FloatTensor(self.saved_qvalues) - torch.FloatTensor(self.saved_state_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + np.finfo(np.float32).eps.item())

        for i in reversed(range(1, len(self.saved_qvalues))):
            advantages[i - 1] += advantages[i]

        signals = returns - advantages

        for r, q, signal, log_prob in zip(returns, self.saved_qvalues, signals, self.saved_logprobs):
            # signal = r - advantage
            policy_losses.append(-signal.detach() * log_prob)
            v_losses.append(F.smooth_l1_loss(q.squeeze(-1), r))

        self.optimizer.zero_grad()

        policy_loss = torch.stack(policy_losses).sum()
        v_loss = torch.stack(v_losses).sum()
        (policy_loss + v_loss).backward()

        self.optimizer.step()

        # optimize the v function
        for i in range(ceil(self.v_update_epochs)):
            old_state_action_pairs, old_values = self.state_RE.sample()
            old_states, old_actions = map(list, zip(*old_state_action_pairs))
            old_states, old_actions, old_values = \
                torch.stack(old_states), torch.stack(old_actions), torch.stack(old_values)

            q_values = self.net.calc_q(old_states, old_actions)
            loss = F.mse_loss(old_values, q_values.squeeze(-1))
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        del self.saved_qvalues[:]
        del self.saved_actions[:]
        del self.saved_logprobs[:]
        del self.saved_rewards[:]
        del self.saved_state_values[:]
        del self.saved_states[:]

        self.state_RE.delete_random()
        self.epsilon_greedy_threshold = max(0, self.epsilon_greedy_threshold - self.epsilon_anneal)
        self.v_update_epochs = max(0, self.v_update_epochs - self.v_update_anneal)
