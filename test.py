import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
from algorithms.a2c import A2C
from algorithms.traj import TrajCVPolicy
from vis import Plotter

env_name = 'CartPole-v0'
algorithm_name = 'a2c'

env = gym.make(env_name)
seed = 23456
env.seed(seed)
torch.manual_seed(seed)
log_interval = 10

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print('Env Name: %s | Seed: %d | State_dim: %d | Action_dim: %d | Algo: %s '
      % (env_name, seed, state_dim, action_dim, algorithm_name))

if algorithm_name == 'a2c':
    model = A2C(state_dim, action_dim)
elif algorithm_name == 'trajcv':
    model = TrajCVPolicy(state_dim, action_dim)
else:
    raise NotImplementedError('Not such algorithm.')

plotter = Plotter(algorithm_name + ' plot', log_interval)


def main():
    running_reward = 0

    for i_episode in range(1000):

        state = env.reset()
        ep_reward = 0

        for t in range(1, 10000):

            action = model.select_action(state)

            state, reward, done, _ = env.step(action)

            model.save_reward(reward)
            model.save_state_action(state, action, done)

            ep_reward += reward

            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        model.finish_episode()
        plotter.add_pair(ep_reward, running_reward)

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     # break

    plotter.show()


if __name__ == '__main__':
    main()
