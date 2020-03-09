import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval

        self.rewards = []
        self.avg_rewards = []

        self.len = 0
        self.graph = None

    def add_pair(self, r, avg_r):
        self.rewards.append(r)
        self.avg_rewards.append(avg_r)
        self.len += 1

    def show(self, stop=-1):
        if stop != -1 and stop < self.len:
            self.len = stop

        x = [i * self.interval for i in range(self.len)]
        plt.plot(x, self.rewards[:self.len])
        plt.plot(x, self.avg_rewards[:self.len])
        plt.legend(['per-epi rewards', 'avg_rewards'])
        plt.title(self.name)
        plt.xlabel('i_episodes')
        plt.ylabel('reward')
        plt.savefig('./plots/single_tests/%s.png' % self.name)