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
env_name = 'CartPole-v0'
gamma = 0.99
lr = 1e-3
v_update_epochs = 30.0
v_update_anneal = 0.0
epsilon_greedy_threshold = 15.0
epsilon_anneal = 0.03
re_sample_batch_size = 100


def worker(worker_id, algorithm_name, seed, return_dict):
    print('Worker %d (pid: %d) has started: algorithm_name <%s> seed <%d>.' % (
        worker_id, os.getpid(), algorithm_name, seed))
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algorithm_name == 'a2c':
        model = A2C(state_dim, action_dim,
                    lr=lr, gamma=gamma,
                    v_update_epochs=v_update_epochs, v_update_anneal=v_update_anneal,
                    epsilon_greedy_threshold=epsilon_greedy_threshold, epsilon_anneal=epsilon_anneal,
                    re_sample_batch_size=re_sample_batch_size)
    elif algorithm_name == 'trajcv':
        model = TrajCVPolicy(state_dim, action_dim,
                             lr=lr, gamma=gamma,
                             v_update_epochs=v_update_epochs, v_update_anneal=v_update_anneal,
                             epsilon_greedy_threshold=epsilon_greedy_threshold, epsilon_anneal=epsilon_anneal,
                             re_sample_batch_size=re_sample_batch_size)
    else:
        raise NotImplementedError('Not such algorithm.')

    reward_records = []

    running_reward = 0

    for i_episode in range(450):

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
    colors = ['C0', 'C1']

    for i in range(2):
        if i == 0:
            rewards_i = np.array(rewards[:len(rewards) // 2])
        else:
            rewards_i = np.array(rewards[len(rewards) // 2:])

        mean = rewards_i.mean(axis=0)
        # mean_itp = interp1d(x, mean, kind='quadratic', fill_value='extrapolate')
        median = np.median(rewards_i, axis=0)

        std = rewards_i.std(axis=0)

        # plt.plot(x, mean_itp(mean), lw=2)
        plt.plot(x, mean, '-', lw=2, color=colors[i])
        plt.plot(x, median, ':', lw=2, color=colors[i])
        # plt.fill_between(x, mean_itp(mean) - std, mean_itp(mean) + std, alpha=0.3)
        plt.fill_between(x, mean - std, mean + std, facecolor=colors[i], alpha=0.3)

    plt.title('Plot of Rewards Averaged over %d Trials' % (len(rewards) // 2))
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.legend(['a2c(mean)', 'a2c(median)', 'trajcv-gae(mean)', 'trajcv-gae(median)'])
    plt.grid()

    plt.savefig('./plots/q/%s_combine.png' % env_name, dpi=200)
    plt.show()


def main():
    manager = mp.Manager()
    return_dict = manager.dict()

    seeds = [1110, 2220, 3330, 4440, 5550] * 2
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
