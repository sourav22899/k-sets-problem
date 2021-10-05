import numpy as np
from sampling import madowSampling

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)


def LMO_A(v, args):
    S_k = np.sort(v)[:args.k].sum()
    z = np.zeros(args.N)
    if S_k > 0:
        return z
    idx = np.argsort(v)[:args.k]
    z[idx] = 1
    return z


def LMO_T(T, v, args):
    x, level = None, 1e10
    for u in T:
        u_ = np.asarray(u)
        if np.inner(v, u_) < level:
            x = u_
            level = np.inner(v, u_)

    return x

def is_in_A(x, args):
    err = 1e-10
    if np.allclose(x, np.zeros(args.N)):
        return True

    if (x >= -err).all() and (x <= 1 + err).all() and x.sum() <= args.k + err:
        return True

    return False


def projection_on_cone(v, args):
    x = np.zeros(args.N)
    T = [list(x)]
    alpha_v, alpha_z, gamma_max = 1, 0, 1
    for i in range(args.n_iters):
        z_t = LMO_A(x - v, args)
        v_t = LMO_T(T, v - x, args)
        T.append(list(z_t))
        d_t = -v_t if np.inner(x - v, z_t) > np.inner(x - v, -v_t) else z_t
        gamma = min(gamma_max, -np.inner(x - v, d_t) / (np.linalg.norm(d_t) ** 2 + 1e-10))
        alpha_v += -gamma
        alpha_z += gamma
        # import pdb; pdb.set_trace()
        if gamma == gamma_max:
            alpha_v = 0
            try:
                T.remove(list(v_t))
            except:
                import pdb; pdb.set_trace()
        gamma_max = alpha_v

        if is_in_A(d_t, args):
            gamma_max = 1e10

        x = x + gamma * d_t

    return x


class BlackWellApproachability:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.k = args.k
        self.f = args.f
        self.eta = args.eta
        self.p = args.p
        self.theta = args.theta
        self.f_list = args.f_list
        self.cnt = args.cnt

    def initialize(self):
        self.p = (self.k / self.N) * np.ones(self.N)
        y_init = np.random.randint(self.N)
        r = np.zeros(self.N)
        r[y_init] = 1
        self.f = r - np.inner(self.p, r)
        self.theta = np.zeros(self.N)
        self.f_list = [self.f]
        self.cnt = 0
        self.eta = 2. / np.sqrt(self.N * np.arange(1, self.args.T + 1))

    def half_space_oracle(self):
        lambda_ = self.theta / (self.theta.sum() + 1e-10)
        threshold = 1 - (self.k - 1) / self.N
        ids = np.where(lambda_ >= threshold)[0]
        m = len(ids)
        gamma = (1. / (self.N - m)) * (self.k - m - 1 + lambda_[ids].sum())
        lambda_ += gamma
        lambda_[ids] = 1.0

        return lambda_

    def blackwell_update_step(self, y):
        # import pdb; pdb.set_trace()

        # OLO update
        theta_ = self.theta + self.eta[self.cnt] * self.f
        theta_ = theta_ / max(1, np.linalg.norm(theta_))
        self.theta = projection_on_cone(theta_, self.args)
        # if not (self.theta >= 0).all():
        #     import pdb; pdb.set_trace()

        # halfspace oracle
        self.p = self.half_space_oracle()

        # receive y and update f for next round
        r = np.zeros(self.N)
        r[y] = 1
        self.f = r - np.inner(self.p, r)
        self.f_list.append(self.f)
        self.cnt += 1


    def get_kset(self, y):
        self.blackwell_update_step(y)
        return self.p, madowSampling(self.N, self.p, self.k)


@ex.automain
def main(_run):
    args = edict(_run.config)
    bwa = BlackWellApproachability(args)
    bwa.initialize()
    for t in range(args.T):
        y = np.random.randint(args.N)
        p, kset = bwa.get_kset(y)
        print(t, y, p.sum(), kset)
        # import pdb;pdb.set_trace()
