import numpy as np


class RE:
    def __init__(self, batch=0):
        self.state_buffer = []
        self.value_buffer = []
        self.size = 0
        self.batch = batch

    def add(self, s, v):
        self.state_buffer.append(s)
        self.value_buffer.append(v)
        self.size += 1

    def sample(self, n=-1):
        if n == -1:
            n = self.batch

        indices = [np.random.randint(0, self.size) for _ in range(n)]
        return list(map(self.state_buffer.__getitem__, indices)), list(map(self.value_buffer.__getitem__, indices))

    def delete(self, n):
        if n >= self.size:
            return
        self.size -= n
        del self.state_buffer[:n]
        del self.value_buffer[:n]

    def delete_half(self):
        self.delete(self.size // 2)

    def delete_random(self, low=-1, high=-1):
        if low >= high or high == -1:
            high = self.size // 2
            low = self.size // 3
        self.delete(np.random.randint(low, high))


