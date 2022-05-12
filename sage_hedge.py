import numpy as np
from sampling import madowSampling
from scipy.special import comb
from decimal import *

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

getcontext().prec = 100

ex = Experiment()
ex = initialise(ex)


class sageHedge:
    def __init__(self, args):
        self.args = args
        self.eta = args.eta
        self.N = args.N
        self.files = np.arange(args.N)
        self.k = args.k
        self.weights = args.weights
        self.R = args.R
        if self.args.method == "iterative":
            self.R = args.R
            self.a = args.a
            self.W = args.W
            self.mul_hedge = Decimal(np.exp(self.eta))
            self.div_hedge = Decimal(np.exp(-self.eta))
            self.mulvec = np.power(self.div_hedge, np.flip(range(self.N - self.k + 1), axis=0))

    def initialize(self):
        self.weights = np.ones(self.N)
        self.R = np.zeros(self.N)
        if self.args.method == "iterative":
            self.W = np.ones((self.N, self.N - self.k + 1), dtype=Decimal)
            one = Decimal(1)
            self.inv_weights = [one for j in range(self.N)]
            self.a = np.zeros(self.N - self.k + 1, dtype=Decimal)
            for j in range(self.N - self.k + 1):
                self.a[j] = (-1) ** (self.N - j) * comb(self.N, j, exact=True)

    def get_normalized_weights(self):
        probs = self.weights / np.sum(self.weights)
        return probs

    def get_kset_direct(self, y):
        onehot_y = np.zeros(self.N)
        onehot_y[y] = 1
        delta = np.exp(self.eta * onehot_y)
        self.weights = self.weights * delta
        K = self.elementary_symmetric_polynomial(self.weights, self.k)
        p = np.zeros(self.N)
        for i in range(self.N):
            W_i = np.delete(self.weights, i)
            p[i] = (self.weights[i] * self.elementary_symmetric_polynomial(W_i, self.k - 1)) / K

        return p, madowSampling(self.N, p, self.k)

    def elementary_symmetric_polynomial(self, X, k):
        X_ = np.zeros(self._next_power_of_2(len(X)), dtype=np.float64)
        X_[:len(X)] = X
        W = np.ones_like(X_, dtype=np.float64)
        X_ = np.vstack((W, X_)).T

        K = X_.shape[0]
        while K > 1:
            X_temp = []
            for i in range(0, K, 2):
                x, y = list(X_[i]), list(X_[i + 1])
                X_temp.append(np.polymul(x, y)[:k + 1])
            X_ = np.asarray(X_temp)
            K = K // 2

        return X_.flatten()[k]

    def get_kset_iterative(self, y):
        p = np.matmul(self.W, self.a) / self.a[self.N - self.k]
        if np.abs(p.sum() - self.k) > 1e-4:
            print(p.sum())

        S = madowSampling(self.N, p, self.k)
        a_new = np.zeros(self.N - self.k + 1, dtype=Decimal)
        a_new[0] = self.mul_hedge * self.a[0]
        for i in range(1, self.N - self.k + 1):
            a_new[i] = self.mul_hedge * self.a[i] + self.inv_weights[y] * (a_new[i - 1] - self.a[i - 1])

        self.inv_weights[y] = self.inv_weights[y] * self.div_hedge
        self.W[y, :] = np.multiply(self.W[y, :], self.mulvec)
        self.a = a_new

        return p, S

    def get_kset_large(self, y):
        # To overcome the numerical precision issues while dealing with large datasets, the following hack
        # is utilised. We use the fact that FTPL with Gumbel Noise is equivalent to Hedge in expectation.
        # Cite: http://proceedings.mlr.press/v35/abernethy14.pdf
        # Although, the equivalence is not exact in our case (as instead of a single expert, we deal with
        # a set of experts), the hack works exceptionally well for practical purposes.

        perturbed_R = self.R + np.random.gumbel(scale=1. / self.eta)
        kset = np.argsort(perturbed_R)[::-1][:self.k]
        self.R[y] += 1
        return None, kset

    def get_kset(self, y):
        p, kset = None, None
        if self.args.method == "large":
            p, kset = self.get_kset_large(y)
        if self.args.method == "iterative":
            p, kset = self.get_kset_iterative(y)
        if self.args.method == "direct":
            p, kset = self.get_kset_direct(y)

        return p, kset

    def _next_power_of_2(self, n):
        count = 0
        if n and not (n & (n - 1)):
            return n
        while n != 0:
            n >>= 1
            count += 1
        return 1 << count


@ex.automain
def main(_run):
    args = edict(_run.config)
    hedge = sageHedge(args)
    hedge.initialize()
    for t in range(args.T):
        y = np.random.randint(args.N)
        p, kset = hedge.get_kset(y)
        if p is None:
            print(t, y, kset)
        else:
            print(t, y, p.sum(), kset)
