import torch
import torch.nn as nn
import torch.nn.functional as F


class PVNet(nn.Module):
    """ Policy network and value network combined """
    def __init__(self, state_dim, action_dim):
        super(PVNet, self).__init__()

        self.affine1 = nn.Linear(state_dim, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

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
