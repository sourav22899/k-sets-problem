import numpy as np

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)


class FTPL:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.k = args.k
        self.R = args.R
        self.eta = args.eta

    def initialize(self):
        self.R = np.zeros(self.N)

    def get_kset(self, y):
        perturbed_R = self.R + self.eta * np.random.standard_normal(self.N)
        kset = np.argsort(perturbed_R)[::-1][:self.k]
        self.R[y] += 1
        return None, kset


@ex.automain
def main(_run):
    args = edict(_run.config)
    ftpl = FTPL(args)
    ftpl.initialize()
    for t in range(args.T):
        y = np.random.randint(args.N)
        p, kset = ftpl.get_kset(y)
        print(t, y, kset)
