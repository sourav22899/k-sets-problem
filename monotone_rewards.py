import pandas as pd
from sage_oco import sageOCOMonotone

import numpy as np
from functools import partial
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
        # top_idx = data["request"].value_counts()[:].index.to_list()
        data = data.loc[data["request"].isin(top_idx)]
    elif args.sample == "random":
        random_idx = np.random.choice(data["request"].max(), size=args.nfiles, replace=False)
        data = data.loc[data["request"].isin(random_idx)]
    elif args.sample == "bottom":
        bottom_idx = data["request"].value_counts()[-args.nfiles:].index.to_list()
        data = data.loc[data["request"].isin(bottom_idx)]

    data['request'] = pd.factorize(data["request"].tolist(), sort=True)[0]

    args.T = len(data)
    args.T_0 = 200
    files = list(data["request"])
    args.N = data["request"].max() + 1
    args.N = args.nfiles
    args.alpha_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    args.algo_list = ["oco"]
    try:
        all_hit_rates = {}
        for alpha in args.alpha_list:
            args.k = int(alpha * args.N)
            print(f"Total files:{args.N}, Timesteps: {args.T}, Cache Size:{args.k}")
            print(f"Max frequency:{data['request'].value_counts().max()}, "
                  f"Min frequency:{data['request'].value_counts().min()}")

            all_algo_rewards = {k: [] for k in args.algo_list}
            for alg in args.algo_list:
                args.algo = alg
                if args.algo == "oco":
                    args.eta = np.sqrt(2 * args.k * np.log(args.N / args.k) / args.T)
                    theory_regret = 2 * np.sqrt(2 * args.k * np.log(args.N / args.k) / np.arange(1, args.T + 1))
                    algo = sageOCOMonotone(args)
                    algo.initialize()
                else:
                    NotImplementedError(f"{args.algo} is not implemented.")

                print('=' * 30, args.algo, '=' * 30)
                print()

                cache = np.random.randint(0, high=args.N, size=args.k)
                total_rewards = 0
                files_seen = np.zeros(args.N)
                regret = []

                if args.resume:
                    log_path = str(args.log_root / f"{args.filename}")
                    with open(log_path, 'rb') as f:
                        algo = pickle.load(f)

                    total_rewards = algo.stats["total_rewards"]
                    files = files[algo.stats["time"] + 1:]
                    files_seen = algo.stats["files_seen"]
                    regret = algo.stats["regret"]

                start_idx = len(regret)
                pbar = tqdm(range(args.T), dynamic_ncols=True, leave=True)
                opt = 0.0

                for t, file in enumerate(files, start=start_idx):
                    raw_reward = 1 - np.abs(np.arange(args.N) - file) / args.N
                    monotone_function = np.max
                    if args.order == 'inf':
                        monotone_function = np.max
                    elif args.order > 0 and args.order == int(args.order):
                        monotone_function = partial(np.linalg.norm, ord=args.order)
                    else:
                        ValueError("Invalid norm order!")
                    # monotone_function = np.max

                    total_rewards += monotone_function(raw_reward[cache]) / monotone_function(raw_reward)
                    opt += (args.k / args.N) * monotone_function(raw_reward)

                    gradient_vector = np.zeros(args.N)
                    gradient_vector[0] = raw_reward[0]
                    for i in range(1, args.N):
                        gradient_vector[i] = monotone_function(raw_reward[:i + 1]) - monotone_function(raw_reward[:i])

                    _, cache = algo.get_kset(grad=gradient_vector)

                    regret.append((opt - total_rewards) / (t + 1))
                    pbar.update(1)
                    pbar.set_description(
                        f"Time: {t + 1} | Total_Reward: {total_rewards} | OPT: {opt} "
                        f"| Actual_Regret:{regret[-1]:4f} | Regret_UB:{theory_regret[t]:4f}"
                    )
                    if t + 1 >= args.T_0:
                        break
                else:
                    NotImplementedError(f"{args.algo} is not implemented.")

                all_algo_rewards[args.algo] = total_rewards
                save_path = str(args.log_root / f"{args.algo}_{get_time()}_alpha={alpha}.pkl")

                with open(save_path, 'wb') as f:
                    stats = {
                        "total_rewards": total_rewards,
                        "files_seen": files_seen,
                        "opt": opt,
                        "time": t,
                        "regret": regret,
                    }
                    algo.stats = stats
                    pickle.dump(algo, f)

                if alg in all_hit_rates.keys():
                    all_hit_rates[alg].append(all_algo_rewards[alg] / args.T_0)
                else:
                    all_hit_rates[alg] = [all_algo_rewards[alg] / args.T_0]

        all_hit_rates['alphas'] = args.alpha_list
        hit_rates = pd.DataFrame(all_hit_rates)
        hit_rates.to_csv(f'{args.dataset}_{args.N}_monotone_rewards_{str(args.order)}_hit_rates.csv', index=False)

    except KeyboardInterrupt:
        save_path = str(args.log_root / f"{args.algo}_{get_time()}_alpha={alpha}.pkl")
        with open(save_path, 'wb') as f:
            stats = {
                "total_rewards": total_rewards,
                "files_seen": files_seen,
                "opt": opt,
                "time": t,
                "regret": regret
            }
            algo.stats = stats
            pickle.dump(algo, f)
