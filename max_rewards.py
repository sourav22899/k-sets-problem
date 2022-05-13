import pandas as pd
from sage_oco import sageOCO

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

def find_opt_cache(args, reward):
    reward_mat = np.zeros((args.nfiles, args.nfiles))
    for i in range(args.nfiles):
        for j in range(args.nfiles):
            if i - j:
                reward_mat[i, j] = max(reward[i], reward[j])

    return reward_mat

ex = Experiment()
ex = initialise(ex)

@ex.automain
def main(_run):
    args = edict(_run.config)
    data = pd.read_csv(f"./data/{args.dataset}_cleaned.csv")
    assert args.dataset in ["wiki", "movielens"]

    if args.sample == "top":
        top_idx = data["request"].value_counts()[args.lim1:args.lim1 + args.nfiles].index.to_list()
        data = data.loc[data["request"].isin(top_idx)]
    elif args.sample == "random":
        random_idx = np.random.choice(data["request"].max(), size=args.nfiles, replace=False)
        data = data.loc[data["request"].isin(random_idx)]
    elif args.sample == "bottom":
        bottom_idx = data["request"].value_counts()[-args.nfiles:].index.to_list()
        data = data.loc[data["request"].isin(bottom_idx)]

    data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]

    args.T = len(data)
    args.T_0 = 7000
    files = list(data["request"])
    args.N = data["request"].max() + 1
    args.N = args.nfiles # set nfiles <= 200
    args.alpha_list = [args.alpha]
    args.algo_list = ["oco"]
    try:
        for alpha in args.alpha_list:
            args.k = int(alpha * args.N)
            print(f"Total files:{args.N}, Timesteps: {args.T}, Cache Size:{args.k}")
            print(f"Max frequency:{data['request'].value_counts().max()}, "
                  f"Min frequency:{data['request'].value_counts().min()}")
            all_algo_regret = {k:[] for k in args.algo_list}
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

                cache = np.random.randint(0, high=args.N, size=args.k)
                total_rewards = 0
                files_seen_histogram = np.zeros(args.N)
                regret = []
                if args.resume:
                    log_path = str(args.log_root / f"{args.filename}")
                    with open(log_path, 'rb') as f:
                        algo = pickle.load(f)

                    total_rewards = algo.stats["total_rewards"]
                    files = files[algo.stats["time"] + 1:]
                    files_seen_histogram = algo.stats["files_seen_histogram"]
                    regret = algo.stats["regret"]

                start_idx = len(regret)
                pbar = tqdm(range(args.T), dynamic_ncols=True, leave=True)
                opt = 0
                cum_reward_mat = np.zeros((args.nfiles, args.nfiles))
                for t, file in enumerate(files, start=start_idx):
                    reward_list = np.abs(np.asarray(cache) - file) / args.N
                    total_rewards += (1 - np.min(reward_list))
                    _, cache = algo.get_kset(file)
                    all_reward = 1 - np.abs(np.arange(args.N) - file) / args.N
                    cum_reward_mat += find_opt_cache(args, all_reward)
                    opt = cum_reward_mat.max()
                    regret.append((opt - total_rewards) / (t+1))
                    pbar.update(1)
                    pbar.set_description(
                        f"Time: {t + 1} | Total_Reward: {total_rewards} | OPT: {opt} "
                        f"| Actual_Regret:{regret[-1]:4f} | Regret_UB:{theory_regret[t]:4f}"
                    )

                    if t + 1 >= args.T_0:
                        break
                else:
                     NotImplementedError(f"{args.algo} is not implemented.")

                all_algo_regret[args.algo] = regret
                save_path = str(args.log_root / f"{args.algo}_{get_time()}_alpha={alpha}.pkl")
                with open(save_path, 'wb') as f:
                    stats = {
                        "total_rewards": total_rewards,
                        "files_seen_histogram": files_seen_histogram,
                        "opt": opt,
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
            lower_bound = 0.02 *  np.sqrt(args.k * np.log(args.N / args.k) / np.arange(1, args.T_0 + 1))
            plt.semilogy(rounds, theory_regret[:args.T_0], '-.', label="regret_upper_bound")
            plt.semilogy(rounds, small_loss_bound, '-.', label="small_loss_bound")
            plt.semilogy(rounds, lower_bound, '-', label="lower_bound")
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
                "opt": opt,
                "time": t,
                "regret": regret
            }
            algo.stats = stats
            pickle.dump(algo, f)
