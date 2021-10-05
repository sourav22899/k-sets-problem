import numpy as np
from timeit import default_timer as timer


def madowSampling(N,
                  p,
                  k):
    """
    :param N:
    :param p:
    :param k:
    :return:

    Assume 0-indexing.
    """
    assert len(p) == N
    S = []
    p = np.insert(p, 0, 0)
    cum_p = np.cumsum(np.asarray(p))
    x = np.random.uniform()
    for i in range(k):
        for j in range(1, len(cum_p)):
            if cum_p[j - 1] <= x + i < cum_p[j]:
                S.append(j - 1)

    return S


if __name__ == "__main__":
    N = 5000
    k = 50
    p = np.random.random(N)
    p = (k / p.sum()) * p
    t = 0.0
    for _ in range(100):
        start = timer()
        S = madowSampling(N, p, k)
        end = timer()
        t += (end - start)
    print(t / 100)
