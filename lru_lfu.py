import numpy as np
from sampling import madowSampling

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)


class LRU:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.k = args.k
        self.cache = args.cache

    def initialize(self):  # randomly initialize a cache configuration
        files = np.random.choice(self.args.N, size=self.args.k, replace=False)
        self.cache = {k: 0 for k in files}

    def update(self, y):
        if y not in self.cache.keys():
            key = max(self.cache, key=self.cache.get)
            self.cache.pop(key)
        for key in self.cache.keys():
            self.cache[key] += 1
        self.cache[y] = 0
        # import pdb; pdb.set_trace()

        if len(self.cache.keys()) != self.args.k:
            import pdb; pdb.set_trace()

    def get_kset(self, y):
        kset = self.cache.keys()
        self.update(y)
        return None, kset


class LFU:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.k = args.k
        self.cache = args.cache

    def initialize(self):  # randomly initialize a cache configuration
        files = np.random.choice(self.args.N, size=self.args.k, replace=False)
        self.cache = {k: 0 for k in files}

    def update(self, y):
        if y not in self.cache.keys():
            key = min(self.cache, key=self.cache.get)
            self.cache.pop(key)
            self.cache[y] = 0
        self.cache[y] += 1
        # import pdb; pdb.set_trace()

        assert len(self.cache.keys()) == self.args.k

    def get_kset(self, y):
        kset = self.cache.keys()
        self.update(y)
        return None, kset


@ex.automain
def main(_run):
    args = edict(_run.config)
    lfu = LFU(args)
    lfu.initialize()
    for t in range(args.T):
        y = np.random.randint(args.N)
        p, kset = lfu.get_kset(y)
        print(t, y, list(kset))
        # import pdb;pdb.set_trace()
