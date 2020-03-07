import torch


def calc_gae(rewards, vs, gamma=0.99, lam=0.95):
    """ calculate the generalized advantage """
    assert (len(rewards) == len(vs) - 1)  # rewards from t = 1 to T, vs from t = 0 to T
    td_res = rewards + gamma * vs[1:] - vs[:-1]
    gae = []
    for t in range(len(td_res)):
        gae.append(td_res[t])
        for l, td in enumerate(td_res[t + 1:]):
            gae[-1] += pow(gamma * lam, l + 1) * td

    return torch.FloatTensor(gae)
