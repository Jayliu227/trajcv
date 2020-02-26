import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
from collections import namedtuple
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

seed = 227
env = gym.make('LunarLander-v2')
env.seed(seed)
torch.manual_seed(seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(8, 128)

        self.action_head = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))

        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_probs, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()
gamma = 0.99
log_interval = 10


def select_action(state):
    state = torch.from_numpy(state).float()
    action_prob, state_value = model.forward(state)

    m = Categorical(action_prob)
    action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.saved_rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, state_value), R in zip(saved_actions, returns):
        advantage = R - state_value.item()

        policy_losses.append(-log_prob * advantage)

        value_losses.append(F.smooth_l1_loss(state_value, torch.tensor([R])))

    optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()

    optimizer.step()

    del model.saved_actions[:]
    del model.saved_rewards[:]


def main():
    running_reward = 10

    for i_episode in count(1):

        state = env.reset()
        ep_reward = 0

        for t in range(1, 10000):

            action = select_action(state)

            state, reward, done, _ = env.step(action)

            model.saved_rewards.append(reward)

            ep_reward += reward

            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        finish_episode()

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
