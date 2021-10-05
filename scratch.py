# import numpy as np
# import scipy.stats as st
#
# def topk_sum(x):
#     return np.sort(x)[::-1][:k].sum()
#
#
# def project(x, k, threshold=1e-10):
#     cnt = 0
#     S = np.sort(x)[::-1][:k].sum()
#     x_orig = x.copy()
#     while S > threshold:
#         idx = np.argsort(x)[::-1][:k]
#         x[idx] = x[idx] - (S / k)
#         cnt += 1
#         S = np.sort(x)[::-1][:k].sum()
#         if cnt > 1e5:
#             import pdb; pdb.set_trace()
#
#     return cnt,  np.linalg.norm(x_orig - x), x_orig, x
#
#
# def thresholding(x, k):
#     idx = np.argsort(x)[::-1][:k]
#     x_orig = x.copy()
#     for i in idx:
#         x[i] = min(0, x[i])
#     return 1, np.linalg.norm(x_orig - x), x_orig, x
#
# T = 100
# limit = 10000
# cnt_max = 0
# kmax = 0
# X, cnt_list = [], []
# for t in range(T):
#     N = np.random.randint(low=2, high=limit)
#     k = max(1, int(0.1 * N))
#     x = np.random.normal(size=N)
#     x_tilda = x.copy()
#     # import pdb; pdb.set_trace()
#     cnt1, norm1, x_orig1, x1 = project(x, k)
#     if cnt1 > cnt_max:
#         cnt_max = max(cnt1, cnt_max)
#         kmax = k
#     # cnt2, norm2, x_orig2, x2 = thresholding(x_tilda, k)
#     cnt2, norm2 = 0, 1e10
#     print(f"t: {t+1}| cnt:{cnt1}, norm1: {norm1:6f} |  cnt:{cnt2}, norm2: {norm2:6f} ")
#     cnt_list.append(cnt1)
#     X.append((N, k))
#     if norm2 < norm1:
#         import pdb; pdb.set_trace()
#
# import pdb;pdb.set_trace()


# cache_comparison.py
""" Comparing the performances of different caching algorithms: MadHedge, FTPL, LRU and LFU
Author: Samrat Mukhopadhyay"""
import pandas as pd
import numpy as np
import scipy as sp
from memoization import cached, CachingAlgorithmFlag as algo
from scipy.special import comb
import pickle
import time
from tqdm import tqdm
from decimal import *

getcontext().prec = 1000
t_count = time.time()


def madow(p, K):
    import numpy as np
    u = np.random.uniform()
    sample = np.ndarray((0, 0))
    a = -u
    for k in range(N):
        b = a
        a = a + float(p[k])
        if np.floor(a) != np.floor(b):
            sample = np.append(sample, int(k))

    return sample


# Extract data from MovieLens dataset, N = 3952
data = pd.read_csv("./ml-latest-small/movielens_cleaned.csv")

files = list(data["request"])
N = data["request"].max() + 1
T = len(files)


# C = alpha * N, alpha typically 1%
alpha = 0.01
C = int(np.floor(N * alpha))

# Write the decorator functions for LRU and LFU:
@cached(max_size=C, algorithm=algo.LRU)
def f_lru(x):
    return x


@cached(max_size=C, algorithm=algo.LFU)
def f_lfu(x):
    return x

D = 0  # Switching cost D
eta_hedge = np.sqrt(C * np.log(N * np.exp(1) / C) / T)

### Constant eta
X = np.zeros((N, T + 1))  # Initialize count vector
R_fixed, R_hedge, R_var, R_lru, R_lfu = [np.zeros((T + 1, 1)) for _ in range(5)]  # Total reward sequence
S_fixed, S_hedge, S_var, S_lru, S_lfu = [np.zeros((T + 1, 1)) for _ in
                                         range(5)]  # Total switching cost sequencesum(a_vec)
Q_fixed, Q_hedge, Q_var, Q_lru, Q_lfu = [np.zeros((T, 1)) for _ in range(5)]  # Regret vector

a_vec = np.zeros((N - C + 1, T + 1), dtype=Decimal)
a_vec[:, 0] = np.array([(-1) ** (N - j) * comb(N, j, exact=True) for j in range(N - C + 1)], dtype=Decimal)
lasty_fixed = lasty_var = np.zeros((N,))

mul_hedge = Decimal(np.exp(eta_hedge))
div_hedge = Decimal(np.exp(-eta_hedge))
one = Decimal(1)
hedge_factor = mul_hedge - one
w = [one for j in range(N)]
W = np.ones((N, N - C + 1), dtype=Decimal)
mulvec = np.power(div_hedge, np.flip(range(N - C + 1), axis=0))

import pdb; pdb.set_trace()

for t in range(T):
    eta_t = np.sqrt(2 * (t + 1) / C) * (4 * np.pi * np.log(N / C)) ** (-1 / 4)

    # p = np.zeros((N, 1))
    # t1 = time.time()
    # for i in range(N):
    #     p[i] = np.dot(W[i, :], a_vec[:, t]) / a_vec[N - C, t]

    # t2 = time.time() - t1
    #
    # print(np.sum(p))
    p = (C/N) * np.ones(N)
    S_t = [int(item) for item in madow(p, C)]

    # Request vector revealed
    x = np.zeros((N,))
    ft = int(files[t]) - 1
    x[ft] = 1
    X[:, t + 1] = X[:, t] + x

    # t3 = time.time()
    # a_vec[0, t + 1] = mul_hedge * a_vec[0, t]
    # for i in range(1, N - C + 1):
    #     a_vec[i, t + 1] = mul_hedge * a_vec[i, t] + w[ft] * (a_vec[i - 1, t + 1] - a_vec[i - 1, t])
    # print(t, time.time() - t_count)

    # Update w vector and W matrix
    w[ft] = w[ft] * div_hedge
    W[ft, :] = np.multiply(W[ft, :], mulvec)

    f_lru(ft)
    f_lfu(ft)

    cache = np.argsort(-X[:, t + 1])
    cache = cache[:C]
    opt = np.sum(X[cache, t + 1])

    # Calculate instantaneous reward of MadHedge
    r_hedge = int(ft in S_t)

    # Update total reward and regret of MadHedge
    R_hedge[t + 1] = R_hedge[t] + r_hedge
    Q_hedge[t] = opt - R_hedge[t + 1]

    R_lru[t + 1], R_lfu[t + 1] = f_lru.cache_info().hits, f_lfu.cache_info().hits
    S_lru[t + 1], S_lfu[t + 1] = f_lru.cache_info().misses, f_lfu.cache_info().misses
    Q_lru[t], Q_lfu[t] = opt - R_lru[t + 1] + S_lru[t + 1], opt - R_lfu[t + 1] + S_lfu[t + 1]

    print(f_lru.cache_info().hits, f_lru.cache_info().misses, t + 1, len(set(files[:t+1])))

    if t % 100 == 0:
        import pdb; pdb.set_trace()


import pdb; pdb.set_trace()
alphastr = str(alpha)
f2 = open('caching_vars_alpha=' + alphastr + '.pickle', 'wb')
variables = dict(Hit=[R_fixed, R_var, R_lru, R_lfu, R_hedge], Switch=[S_fixed, S_var, S_lru, S_lfu],
                 Regret=[Q_fixed, Q_var, Q_lru, Q_lfu, Q_hedge], C=C, N=N, T=T)
pickle.dump(variables, f2)
