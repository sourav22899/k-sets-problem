import numpy as np
from sampling import madowSampling

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

    def initialize(self):
        self.R = np.zeros(self.N)

    def calculate_marginals_oco(self, y, vec=None):
        R_ = np.sort(self.R)[::-1]
        exp_R = np.exp(self.eta * (R_))
        tailsum_expR = np.cumsum(exp_R)[::-1]  # tailsum
        i_star = 0
        for i in range(self.k):
            if (self.k - i) * exp_R[i] >= (tailsum_expR[i]):
                i_star = i

        K = (self.k - i_star) / (tailsum_expR[i_star])
        p = np.minimum(np.ones(self.N), K * np.exp(self.eta * (self.R)))
        if np.abs(p.sum() - self.k) > 1e-3:
            print(p.sum())

        # gradient update step
        self.R[y] += 1  # for all the experiments except monotone set function.

        return p

    def get_kset(self, y):
        p = self.calculate_marginals_oco(y)
        return p, madowSampling(self.N, p, self.k)


class sageOCOMonotone:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.k = args.k
        self.R = args.R
        self.eta = args.eta

    def initialize(self):
        self.R = np.zeros(self.N)

    def calculate_marginals_oco(self, grad):
        R_ = np.sort(self.R)[::-1]
        exp_R = np.exp(self.eta * R_)
        tailsum_expR = np.cumsum(exp_R)[::-1]  # tailsum
        i_star = 0
        for i in range(self.k):
            if (self.k - i) * exp_R[i] >= (tailsum_expR[i]):
                i_star = i

        K = (self.k - i_star) / (tailsum_expR[i_star])
        p = np.minimum(np.ones(self.N), K * np.exp(self.eta * (self.R)))
        if np.abs(p.sum() - self.k) > 1e-3:
            print(p.sum())

        # gradient update step
        self.R += grad

        return p

    def get_kset(self, grad):
        p = self.calculate_marginals_oco(grad)
        return p, madowSampling(self.N, p, self.k)


@ex.automain
def main(_run):
    args = edict(_run.config)
    oco = sageOCOMonotone(args)
    oco.initialize()
    for t in range(args.T):
        y = np.random.randint(args.N)
        p, kset = oco.get_kset(y)
        print(t, y, p.sum(), kset)
