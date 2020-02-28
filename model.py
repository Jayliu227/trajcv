import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        s = F.softmax(self.affine3(s))
        return s


class TrajCVPolicy:
    def __init__(self, state_dim, action_dim, ):
        self.policy = Policy(state_dim, action_dim)
        self.Q = QFunc(state_dim)

        self.saved_actions = []
        self.saved_rewards = []

    def select_action(self):
        pass

    def finish_episode(self):
        pass


env = gym.make('CartPole-v0')
state = env.reset()
action = 1
state, reward, done, _ = env.step(action)

state = torch.FloatTensor(state)

p = Policy(4, 2)
p.forward(state)
