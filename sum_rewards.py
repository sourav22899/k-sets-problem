import pandas as pd
from sage_hedge import sageHedge
from ftpl import FTPL
from memoization import cached, CachingAlgorithmFlag as algorithms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict
from datetime import datetime

def get_time():
    now = datetime.now()
    str = [now.strftime("%Y"), now.strftime("%h"), now.strftime("%d"),
           now.strftime("%H"), now.strftime("%M"), now.strftime("%S")]
    str = '_'.join(str)
    return str

ex = Experiment()
ex = initialise(ex)

@ex.automain
def main(_run):
    args = edict(_run.config)
    data = pd.read_csv(f"./data/{args.dataset}_cleaned.csv")
    assert args.dataset in ["wiki", "movielens"]
    if args.sample == "all":
        top_idx = data["request"].value_counts()[:].index.to_list()
        data = data.loc[data["request"].isin(top_idx)]
    elif args.sample == "random":
        random_idx = np.random.choice(data["request"].max(), size=args.nfiles, replace=False)
        data = data.loc[data["request"].isin(random_idx)]
    elif args.sample == "bottom":
        bottom_idx = data["request"].value_counts()[-args.nfiles:].index.to_list()
        data = data.loc[data["request"].isin(bottom_idx)]

    data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]

    args.T = len(data)
    args.T_0 = args.T
    files = list(data["request"])
    args.N = data["request"].max() + 1
    try:
        for alpha in args.alpha_list:
            args.k = int(alpha * args.N)
            print(f"Total files:{args.N}, Timesteps: {args.T}, Cache Size:{args.k}")
            print(f"Max frequency:{data['request'].value_counts().max()}, "
                  f"Min frequency:{data['request'].value_counts().min()}")

            all_algo_regret = {k:[] for k in args.algo_list}
            for alg in args.algo_list:
                args.algo = alg
                if args.algo == "hedge":
                    args.eta = np.sqrt(args.k * np.log(args.N * np.exp(1) / args.k) / args.T)
                    theory_regret = np.sqrt(2 * args.k * np.log(args.N * np.exp(1) / args.k) / np.arange(1, args.T + 1))
                    algo = sageHedge(args)
                    algo.initialize()
                elif args.algo == "ftpl":
                    args.eta = ((4 * np.pi * np.log(args.N)) ** (-0.25)) * np.sqrt(args.T / args.k)
                    theory_regret = 1.51 * np.sqrt(np.sqrt(np.log(args.N)) * args.k / np.arange(1, args.T + 1))
                    algo = FTPL(args)
                    algo.initialize()
                else:
                    NotImplementedError(f"{args.algo} is not implemented.")

                print('='*30, args.algo, '='*30)
                print()

                if args.algo not in ["lru", "lfu"]:
                    cache = np.random.randint(0, high=args.N, size=args.k)
                total_rewards = 0
                files_seen_histogram = np.zeros(args.N)
                regret = []

                if args.resume:
                    assert args.algo not in ["lru", "lfu"]
                    log_path = str(args.log_root / f"{args.filename}")
                    with open(log_path, 'rb') as f:
                        algo = pickle.load(f)
                    total_rewards = algo.stats["total_rewards"]
                    files = files[algo.stats["time"] + 1:]
                    files_seen_histogram = algo.stats["files_seen_histogram"]
                    regret = algo.stats["regret"]

                start_idx = len(regret)
                pbar = tqdm(range(args.T), dynamic_ncols=True, leave=True)
                if args.algo not in ["lru", "lfu"]:
                    for t, file in enumerate(files, start=start_idx):
                        if file in cache:
                            total_rewards += 1
                        _, cache = algo.get_kset(file)

                        files_seen_histogram[file] += 1
                        opt = files_seen_histogram[(-files_seen_histogram).argsort()[:args.k]].sum()
                        regret.append((opt - total_rewards) / (t + 1))
                        pbar.update(1)
                        pbar.set_description(
                            f"Time: {t + 1} | Total_Reward: {total_rewards} | OPT: {opt} "
                            f"| Actual_Regret:{regret[-1]:4f} | Regret_UB:{theory_regret[t]:4f}"
                        )
                        if t + 1 >= args.T_0:
                            break

                else:
                    @cached(max_size=args.k, algorithm=algorithms.LRU)
                    def f_lru(x):
                        return x

                    @cached(max_size=args.k, algorithm=algorithms.LFU)
                    def f_lfu(x):
                        return x

                    cache_algo = f_lfu if args.algo == "lfu" else f_lru

                    for t, file in enumerate(files, start=start_idx):
                        cache_algo(file)
                        total_rewards = cache_algo.cache_info().hits
                        files_seen_histogram[file] += 1
                        opt = files_seen_histogram[(-files_seen_histogram).argsort()[:args.k]].sum()
                        regret.append((opt - total_rewards) / (t + 1))
                        pbar.update(1)
                        pbar.set_description(
                            f"Time: {t + 1} | Total_Reward: {total_rewards} | OPT: {opt} "
                            f"| Actual_Regret:{regret[-1]:4f} | Regret_UB:{theory_regret[t]:4f}"
                        )
                        if t + 1 >= args.T_0:
                            break

                all_algo_regret[args.algo] = regret
                save_path = str(args.log_root / f"{args.algo}_{get_time()}_alpha={alpha}.pkl")
                with open(save_path, 'wb') as f:
                    stats = {
                        "total_rewards": total_rewards,
                        "files_seen_histogram": files_seen_histogram,
                        "time": t,
                        "regret": regret,
                    }
                    algo.stats = stats
                    pickle.dump(algo, f)

            rounds = np.arange(1, args.T_0 + 1)
            plt.figure(figsize=(9, 6))
            plt.grid(linestyle='dashed', which='both')
            linestyles = ['-', '--', '-.', ':']
            for i, alg in enumerate(all_algo_regret.keys()):
                plt.semilogy(rounds, all_algo_regret[alg],
                             linestyle=linestyles[i % 4], label=f"{alg.upper()}",
                             antialiased=True)

            # theory_regret for Hedge.
            theory_regret = np.sqrt(2 * args.k * np.log(args.N * np.exp(1) / args.k) / np.arange(1, args.T + 1))
            small_loss_bound = theory_regret[:args.T_0] * np.sqrt(opt / np.arange(1, args.T_0 + 1))
            plt.semilogy(rounds, theory_regret[:args.T_0], '-.', label="regret_upper_bound")
            plt.semilogy(rounds, small_loss_bound, '-.', label="small_loss_bound")
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
                "files_seen_histogram": files_seen_histogram,
                "time": t,
                "regret": regret
            }
            algo.stats = stats
            pickle.dump(algo, f)
