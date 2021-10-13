import numpy as np
from sampling import madowSampling
from decimal import Decimal, Context
from math import *

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)


class sageOCO:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.k = args.k
        self.R = args.R
        self.eta = args.eta
        # self.eta = Decimal(self.eta)

    def initialize(self):
        self.R = np.zeros(self.N)
        # self.R = np.zeros(self.N, dtype=Decimal)

    # def calculate_marginals_oco(self, y):
    #     one = Decimal(1)
    #     e = one.exp()
    #     # import pdb; pdb.set_trace()
    #     R_ = np.sort(self.R)[::-1]
    #     exp_R = []
    #     ctx = Context()
    #     for i in range(R_.shape[0]):
    #         exp_R.append(ctx.power(e, self.eta * (R_[i] - R_.max())))
    #     exp_R = np.asarray(exp_R, dtype=Decimal)
    #     tailsum_expR = np.cumsum(exp_R)[::-1]  # tailsum
    #     i_star = 0
    #     for i in range(self.k):
    #         if (self.k - i) * exp_R[i] >= (tailsum_expR[i]):
    #             i_star = i
    #
    #     K = Decimal(self.k - i_star) / (tailsum_expR[i_star])
    #     array1 = np.asarray(np.ones(self.N), dtype=Decimal)
    #     array2 = []
    #     for i in range(R_.shape[0]):
    #         array2.append(K * (self.eta * (self.R[i] - self.R.max())).exp())
    #     array2 = np.asarray(array2, dtype=Decimal)
    #     p = np.minimum(array1, array2)
    #     p = np.asarray([Decimal(x) for x in p], dtype=Decimal)
    #     if np.abs(p.sum() - self.k) > 1e-3:
    #         print(p.sum())
    #         import pdb; pdb.set_trace()
    #
    #     self.R[y] += one
    #     return p

    def calculate_marginals_oco(self, y):
        # import pdb; pdb.set_trace()
        R_ = np.sort(self.R)[::-1]
        exp_R = np.exp(self.eta * (R_))
        tailsum_expR = np.cumsum(exp_R)[::-1]  # tailsum
        i_star = 0
        for i in range(self.k):
            if (self.k - i) * exp_R[i] >= (tailsum_expR[i]):
                i_star = i

        K = (self.k - i_star) / (tailsum_expR[i_star])
        p = np.minimum(np.ones(self.N), K * np.exp(self.eta * (self.R)))
        # print(self.R.dtype)
        if np.abs(p.sum() - self.k) > 1e-3:
            print(p.sum())
            # import pdb; pdb.set_trace()
        # print(p.sum(), exp_R.max(), exp_R.min())

        self.R[y] += 1
        return p

    def get_kset(self, y):
        p = self.calculate_marginals_oco(y)
        return p, madowSampling(self.N, p, self.k)

    # def get_kset(self, y):
    #     perturbed_R = self.R + np.random.exponential(scale=1. / self.eta)
    #     kset = np.argsort(perturbed_R)[::-1][:self.k]
    #     self.R[y] += 1
    #     return None, kset


@ex.automain
def main(_run):
    args = edict(_run.config)
    oco = sageOCO(args)
    oco.initialize()
    for t in range(args.T):
        y = np.random.randint(args.N)
        p, kset = oco.get_kset(y)
        print(t, y, p.sum(), kset)
        # import pdb;pdb.set_trace()
