import scipy.signal


def calc_gae(rewards, vs, gamma, lam):
    """ calculate the generalized advantage """
    pass

# gamma = .2349
# x = [1., 21., 1., 13., 11.]
#
# ans = []
# for i in range(len(x)):
#     ans.append(x[i])
#     for p, k in enumerate(x[i+1:]):
#         ans[-1] += pow(gamma, p + 1) * k
#
# print(ans)
#
# print(scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1])
