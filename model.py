import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical
from itertools import count


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

        self.policy = Policy(state_dim, action_dim)
        self.Q = QFunc(state_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.saved_logprobs = []
        self.saved_qvalues = []
        self.saved_expectedqs = []
        self.saved_rewards = []

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
        self.saved_qvalues.append(all_qvalues[a.item()])
        # save the log probability of the action
        self.saved_logprobs.append(m.log_prob(a))

        return a.item()

    def save_reward(self, r):
        self.saved_rewards.append(r)

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

        del self.saved_logprobs[:]
        del self.saved_qvalues[:]
        del self.saved_expectedqs[:]
        del self.saved_rewards[:]


def main():
    seed = 227
    env = gym.make('CartPole-v0')
    env.seed(seed)
    torch.manual_seed(seed)
    log_interval = 10

    trajcv = TrajCVPolicy(4, 2)

    running_reward = 10

    for i_episode in count(1):

        state = env.reset()
        ep_reward = 0

        for t in range(1, 10000):

            action = trajcv.select_action(state)

            state, reward, done, _ = env.step(action)

            trajcv.save_reward(reward)

            ep_reward += reward

            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        trajcv.finish_episode()

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
