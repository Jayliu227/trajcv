import scipy.signal


def calc_gae(rewards, vs, gamma=0.99, lam=0.95):
    """ calculate the generalized advantage """
    assert (len(rewards) == len(vs) - 1)  # rewards from t = 1 to T, vs from t = 0 to T
    td_res = rewards + gamma * vs[1:] - vs[:-1]
    return scipy.signal.lfilter([1], [1, -gamma * lam], td_res[::-1], axis=0)[::-1]
