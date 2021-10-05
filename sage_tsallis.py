# cache_comparison.py
""" Comparing the performances of different caching algorithms: MadHedge, FTPL, LRU and LFU
Author: Samrat Mukhopadhyay"""
import pandas as pd
import numpy as np
import scipy as sp
from memoization import cached, CachingAlgorithmFlag as algo
import pickle
import time
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
# data = pd.read_csv("CMU_dataset.csv")
#
# enc_files = data.T.iloc[1]  # Encoded File indices in chronological order
# uniqueIds = set(enc_files)
# N = len(uniqueIds)
# ids = range(1, N + 1)
# filemap = {k: v for (k, v) in zip(uniqueIds, ids)}
# files = [filemap[x] for x in enc_files]
#
# # N = 100
# testfiles = []
# for i in range(len(files)):
#     if files[i] <= N:
#         testfiles.append(files[i])
# files = testfiles
# T = len(files)
T = 1000
files = np.random.randint(1, N, size=T)


# C = alpha * N, alpha typically 1%
alpha = 0.05
C = int(np.floor(N * alpha))


# Write the decorator functions for LRU and LFU:
@cached(max_size=C, algorithm=algo.LRU)
def f_lru(x):
    return x


@cached(max_size=C, algorithm=algo.LFU)
def f_lfu(x):
    return x


# Run FTPL with static learning rate  
D = 0  # Switching cost D
eta_fixed = np.sqrt(2 * T / C) * (4 * np.pi * np.log(N / C)) ** (
            -1 / 4)  # eta = sqrt(T(D+1)/C)(4 * pi * ln(N/C))^(-1/4)
eta_hedge = np.sqrt(C * np.log(N * np.exp(1) / C) / T)
gamma = np.random.randn(N, )  # Sample the perturbation from standard Gaussian distribution

### Constant eta
X = np.zeros((N, T + 1))  # Initialize count vector
R_fixed, R_hedge, R_var, R_lru, R_lfu = [np.zeros((T + 1, 1)) for _ in range(5)]  # Total reward sequence
S_fixed, S_hedge, S_var, S_lru, S_lfu = [np.zeros((T + 1, 1)) for _ in
                                         range(5)]  # Total switching cost sequencesum(a_vec)
Q_fixed, Q_hedge, Q_var, Q_lru, Q_lfu = [np.zeros((T, 1)) for _ in range(5)]  # Regret vector

a_vec = np.zeros((N - C + 1, T + 1), dtype=Decimal)
a_vec[:, 0] = np.array([(-1) ** (N - j) * sp.special.comb(N, j, exact=True) for j in range(N - C + 1)], dtype=Decimal)
lasty_fixed = lasty_var = np.zeros((N,))

mul_hedge = Decimal(np.exp(eta_hedge))
div_hedge = Decimal(np.exp(-eta_hedge))
one = Decimal(1)
hedge_factor = mul_hedge - one
w = [one for j in range(N)]
W = np.ones((N, N - C + 1), dtype=Decimal)
mulvec = np.power(div_hedge, np.flip(range(N - C + 1), axis=0))


for t in range(T):
    #    if int(files[t]) <= N:
    eta_t = np.sqrt(2 * (t + 1) / C) * (4 * np.pi * np.log(N / C)) ** (-1 / 4)
    #    print('time=', t)
    # X_per is the perturbed count vector
    X_per_fixed = X[:, t] + eta_fixed * gamma
    X_per_var = X[:, t] + eta_t * gamma

    # The FTPL predicted cache configuration
    y_fixed, y_var = np.zeros((N,)), np.zeros((N,))
    y_fixed[np.argsort(-X_per_fixed)[:C]] = 1
    y_var[np.argsort(-X_per_var)[:C]] = 1

    p = np.zeros((N, 1))
    t1 = time.time()
    for i in range(N):
        p[i] = np.dot(W[i, :], a_vec[:, t]) / a_vec[N - C, t]

    t2 = time.time() - t1

    print(np.sum(p))
    S_t = [int(item) for item in madow(p, C)]

    # Request vector revealed
    x = np.zeros((N,))
    ft = int(files[t]) - 1
    x[ft] = 1
    X[:, t + 1] = X[:, t] + x

    t3 = time.time()
    a_vec[0, t + 1] = mul_hedge * a_vec[0, t]
    for i in range(1, N - C + 1):
        a_vec[i, t + 1] = mul_hedge * a_vec[i, t] + w[ft] * (a_vec[i - 1, t + 1] - a_vec[i - 1, t])
    print(t, time.time() - t_count)

    # Update w vector and W matrix
    w[ft] = w[ft] * div_hedge
    W[ft, :] = np.multiply(W[ft, :], mulvec)

    # Update cache using LFU, LRU and FIFO
    f_lru(ft)
    f_lfu(ft)

    # Calculate instantaneous reward and switching cost for FTPL
    r_fixed = y_fixed[ft]
    r_var = y_var[ft]
    if t == 0:
        s_fixed, s_var = 0, 0
    else:
        s_fixed = 0.5 * D * np.sum(np.abs(y_fixed - lasty_fixed))
        s_var = 0.5 * D * np.sum(np.abs(y_var - lasty_var))

    # Calculate instantaneous reward of MadHedge
    r_hedge = int(ft in S_t)

    # Update total reward, switching cost and regret for FTPL
    lasty_fixed = y_fixed
    lasty_var = y_var
    cache = np.argsort(-X[:, t + 1])
    cache = cache[:C]
    opt = np.sum(X[cache, t + 1])
    # Reward, Switching Cost and Regret update for FTPL
    R_fixed[t + 1], R_var[t + 1] = R_fixed[t] + r_fixed, R_var[t] + r_var
    S_fixed[t + 1], S_var[t + 1] = S_fixed[t] + s_fixed, S_var[t] + s_var
    Q_fixed[t], Q_var[t] = opt - R_fixed[t + 1] + S_fixed[t + 1], opt - R_var[t + 1] + S_var[t + 1]

    # Update total reward and regret of MadHedge
    R_hedge[t + 1] = R_hedge[t] + r_hedge
    Q_hedge[t] = opt - R_hedge[t + 1]

    # Total Regret and Switching Cost update for LRU, LFU and FIFO
    R_lru[t + 1], R_lfu[t + 1] = f_lru.cache_info().hits, f_lfu.cache_info().hits
    S_lru[t + 1], S_lfu[t + 1] = f_lru.cache_info().misses, f_lfu.cache_info().misses
    Q_lru[t], Q_lfu[t] = opt - R_lru[t + 1] + S_lru[t + 1], opt - R_lfu[t + 1] + S_lfu[t + 1]

alphastr = str(alpha)
f2 = open('caching_vars_alpha=' + alphastr + '.pickle', 'wb')
variables = dict(Hit=[R_fixed, R_var, R_lru, R_lfu, R_hedge], Switch=[S_fixed, S_var, S_lru, S_lfu],
                 Regret=[Q_fixed, Q_var, Q_lru, Q_lfu, Q_hedge], C=C, N=N, T=T)
pickle.dump(variables, f2)
