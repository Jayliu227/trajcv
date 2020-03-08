import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from scipy.interpolate import interp1d

from algorithms.traj import TrajCVPolicy
from algorithms.a2c import A2C

log_interval = 10
lr = 1e-3


def worker(worker_id, algorithm_name, seed, return_dict):
    print('Worker %d (pid: %d) has started: algorithm_name <%s> seed <%d>.' % (
        worker_id, os.getpid(), algorithm_name, seed))
    env = gym.make('CartPole-v0')
    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algorithm_name == 'a2c':
        model = A2C(state_dim, action_dim, lr=lr)
    elif algorithm_name == 'trajcv':
        model = TrajCVPolicy(state_dim, action_dim, lr=lr)
    else:
        raise NotImplementedError('Not such algorithm.')

    reward_records = []

    running_reward = 0

    for i_episode in range(100):

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
        reward_records.append(running_reward)

        model.finish_episode()

        if i_episode % log_interval == 0:
            print('{:>10}(pid:{:>4}-worker_id:{:>2})|Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                algorithm_name, seed, worker_id, i_episode, ep_reward, running_reward))

    env.close()
    return_dict[worker_id] = reward_records
    print('Worker %d has ended.' % worker_id)


def plot(rewards):
    x = np.array([i for i in range(len(rewards[0]))])

    for i in range(2):
        if i == 0:
            rewards_i = np.array(rewards[:len(rewards) // 2])
        else:
            rewards_i = np.array(rewards[len(rewards) // 2:])

        mean = rewards_i.mean(axis=0)
        mean_itp = interp1d(x, mean, kind='quadratic', fill_value='extrapolate')

        std = rewards_i.std(axis=0)

        plt.plot(x, mean_itp(mean), lw=2)
        plt.fill_between(x, mean_itp(mean) - std, mean_itp(mean) + std, alpha=0.3)

    plt.title('Plot of Rewards Averaged over %d Trials' % (len(rewards) // 2))
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.legend(['a2c', 'trajcv-gae'])
    plt.grid()

    plt.savefig('./combine.png')
    plt.show()


def main():
    manager = mp.Manager()
    return_dict = manager.dict()

    seeds = [1023, 1234, 2345, 3456, 4567] * 2
    names = []
    for i in range(5):
        names.append('trajcv')
        names.insert(0, 'a2c')

    processes = []
    for i in range(10):
        p = mp.Process(target=worker, args=(i, names[i], seeds[i], return_dict))
        processes.append(p)
        p.start()

    for i in processes:
        i.join()

    result = [return_dict[key] for key in sorted(return_dict.keys(), reverse=False)]
    plot(result)


if __name__ == '__main__':
    main()
