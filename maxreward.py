import pandas as pd
from sage_hedge import sageHedge
from sage_oco import sageOCO
from ftpl import FTPL
from lru_lfu import LRU, LFU
from blackwell import BlackWellApproachability
from memoization import cached, CachingAlgorithmFlag as algorithms



import numpy as np
import matplotlib.pyplot as plt
from decimal import *
from tqdm import tqdm
import pickle

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict
from datetime import datetime

#plt.style.use('ggplot')

def get_time():
    now = datetime.now()
    str = [now.strftime("%Y"), now.strftime("%h"), now.strftime("%d"),
           now.strftime("%H"), now.strftime("%M"), now.strftime("%S")]
    str = '_'.join(str)
    return str

def min_l2dist(file, cache):
    idx = -1
    dist = 1e7
    for exp in cache.keys():
        if np.linalg.norm(cache[exp] - file) < dist:
            dist = np.linalg.norm(cache[exp] - file)
            idx = exp

    return idx, 1 - np.exp(-dist / file.shape[0])

getcontext().prec = 2000 ##


ex = Experiment()
ex = initialise(ex)

@ex.automain
def main(_run):
    args = edict(_run.config)
    data = np.load('./data/mnist.npy')
    experts = np.load('./data/experts.npy')
    expert_dict = {}
    for i in range(experts.shape[0]):
        expert_dict[i] = experts[i]
    args.T = data.shape[0]
    args.T_0 = args.T
    files = data
    args.N = experts.shape[0]
    args.alpha_list = [0.1]
    args.algo_list = ["oco"]
    try:
        for alpha in args.alpha_list:
            args.k = int(alpha * args.N)
            print(f"Total Sites:{args.N}, Timesteps: {args.T}, Cluster Size:{args.k}")
            # print(f"Max frequency:{data['request'].value_counts().max()}, "
            #       f"Min frequency:{data['request'].value_counts().min()}")
            import pdb;pdb.set_trace()
            all_algo_regret = {k:[] for k in args.algo_list}
            # import  pdb; pdb.set_trace()
            for alg in args.algo_list:
                args.algo = alg
                if args.algo == "oco":
                    args.eta = np.sqrt(2 * args.k * np.log(args.N / args.k) / args.T)
                    theory_regret = 2 * np.sqrt(2 * args.k * np.log(args.N / args.k) / np.arange(1, args.T + 1))
                    algo = sageOCO(args)
                    algo.initialize()
                else:
                    NotImplementedError(f"{args.algo} is not implemented.")

                print('='*30, args.algo, '='*30)
                print()

                if args.algo not in ["lru", "lfu"]:
                    cache_ids = np.random.randint(0, high=args.N, size=args.k)
                    cache = dict((i, expert_dict[i]) for i in cache_ids)
                total_rewards = 0
                opt_cost = 0
                regret = []

                if args.resume:
                    assert args.algo not in ["lru", "lfu"]
                    log_path = str(args.log_root / f"{args.filename}")
                    with open(log_path, 'rb') as f:
                        algo = pickle.load(f)

                    total_rewards = algo.stats["total_rewards"]
                    files = files[algo.stats["time"] + 1:]
                    opt_cost = algo.stats["opt_cost"]
                    regret = algo.stats["regret"]

                # import pdb; pdb.set_trace()
                start_idx = len(regret)

                pbar = tqdm(range(args.T), dynamic_ncols=True, leave=True)

                if args.algo not in ["lru", "lfu"]:
                    for t, file in enumerate(files, start=start_idx):
                        # import pdb; pdb.set_trace()
                        idx, dist = min_l2dist(file, cache)
                        total_rewards += dist
                        _, cache_ids = algo.get_kset(idx)
                        cache = dict((i, expert_dict[i]) for i in cache_ids)
                        # print(p.sum())
                        k_ = t // 5000
                        opt_experts = dict((i, expert_dict[i]) for i in range(k_, k_ + args.k))

                        opt_cost += min_l2dist(file, opt_experts)[1]
                        regret.append((- opt_cost + total_rewards) / (t+1))
                        pbar.update(1)
                        pbar.set_description(
                            f"Time: {t + 1} | Total_Reward: {total_rewards} | OPT: {opt_cost} "
                            f"| Actual_Regret:{regret[-1]:4f} | Regret_UB:{theory_regret[t]:4f}"
                        )
                        # if t % 100 == 0:
                        #     import pdb; pdb.set_trace()

                        if t + 1 >= args.T_0:
                            break

                # import pdb; pdb.set_trace()
                all_algo_regret[args.algo] = regret

                save_path = str(args.log_root / f"{args.algo}_{get_time()}_alpha={alpha}.pkl")

                with open(save_path, 'wb') as f:
                    stats = {
                        "total_rewards": total_rewards,
                        "opt_cost": opt_cost,
                        "time": t,
                        "regret": regret,
                    }
                    algo.stats = stats
                    pickle.dump(algo, f)

            rounds = np.arange(1, args.T_0 + 1)
            T_i = 5000

            plt.figure(figsize=(9, 6))
            plt.grid(linestyle='dashed', which='both')
            linestyles = ['-', '--', '-.', ':']
            for i, alg in enumerate(all_algo_regret.keys()):
                plt.semilogy(rounds, all_algo_regret[alg],
                             linestyle=linestyles[i % 4], label=f"{alg.upper()}",
                             antialiased=True)

            # theory_regret for Hedge.
            theory_regret = np.sqrt(2 * args.k * np.log(args.N * np.exp(1) / args.k) / np.arange(1, args.T + 1))
            # theory_regret = 1.51 * np.sqrt(np.sqrt(np.log(args.N)) * args.k / np.arange(1, args.T + 1))
            small_loss_bound = theory_regret[:args.T_0] * np.sqrt(opt_cost / np.arange(1, args.T_0 + 1))
            lower_bound = 0.25 * np.exp(1) * np.sqrt(args.k * np.log(args.N / args.k) / np.arange(1, args.T_0 + 1))
            lower_bound = lower_bound - 1 * args.k ** 3 / (args.N * np.arange(1, args.T_0 + 1))
            lower_bound = np.maximum(1e-6 * np.ones(args.T_0), lower_bound)
            plt.semilogy(rounds[T_i:], theory_regret[T_i:args.T_0], '-.', label="regret_upper_bound")
            plt.semilogy(rounds[T_i:], small_loss_bound[T_i:], '-.', label="small_loss_bound")
            plt.semilogy(rounds[T_i:], lower_bound[T_i:], '-', label="lower_bound")
            plt.ylabel(r"$R_T/T$")
            plt.xlabel(r"T")
            plt.legend()
            filename =  '_'.join(args.algo_list) + f'_{get_time()}_N={args.N}_alpha={alpha}.png'
            save_path = str(args.fig_root / filename)
            plt.savefig(save_path)

    except KeyboardInterrupt:
        save_path = str(args.log_root / f"{args.algo}_{get_time()}_alpha={alpha}.pkl")
        with open(save_path, 'wb') as f:
            stats = {
                "total_rewards": total_rewards,
                "opt_cost": opt_cost,
                "time": t,
                "regret": regret
            }
            algo.stats = stats
            pickle.dump(algo, f)
