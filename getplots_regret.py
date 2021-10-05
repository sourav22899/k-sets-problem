import seaborn as sns
from glob import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['text.usetex'] = True

from sacred import Experiment
from config import initialise
from easydict import EasyDict as edict

ex = Experiment()
ex = initialise(ex)

@ex.automain
def main(_run):
    args = edict(_run.config)
    args.N = 180
    alpha = '0.1'
    args.k = int(float(alpha) * args.N)


    files = glob(f'../hpce_results/logs/*{alpha}.pkl')
    print(files)

    all_regret_dict = {}
    for file in files:
        if file.find('Oct_04') != -1:
            print(file)
            alg_name = file.split('/')[-1].split('_')[0]
            with open(file, 'rb') as f:
                algo = pickle.load(f)
                all_regret_dict[alg_name.upper()] = algo.stats["regret"]
                args.T = algo.stats['time'] + 1
                opt = int(args.T * algo.stats['regret'][-1] + algo.stats['total_rewards'])


            # import pdb; pdb.set_trace()

    theory_regret = np.sqrt(2 * args.k * np.log(args.N * np.exp(1) / args.k) / np.arange(1, args.T + 1))
    # sns.set_style('whitegrid')
    sns.set_style("darkgrid")
    # sns.set_context("poster")
    # sns.color_palette("gist_heat")
    sns.color_palette("bright")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})

    all_regret_dict['Regret Upper Bound'] = theory_regret
    all_regret_dict['Small Loss Bound'] = theory_regret * np.sqrt(opt / np.arange(1, args.T + 1))
    # plt.semilogy(all_regret_dict['OCO'])
    # plt.show()

    data = pd.DataFrame(all_regret_dict)
    print(data.head())
    f, ax = plt.subplots()
    g_results = sns.lineplot(data=data)
    ax.set(yscale="log")
    g_results.set_xlabel("Time", fontsize=15)
    g_results.set_ylabel("Time-averaged Regret", fontsize=15)
    plt.grid(linestyle='dashed', which='both')
    plt.savefig(f"n={args.N}_alpha={alpha}.svg", format="svg")
    # locmin = mticker.LogLocator(base=10, subs=np.arange(0.1, 1, 0.1), numticks=10)
    # g_results.ax.yaxis.set_minor_locator(locmin)
    # g_results.ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # fig, ax = plt.subplots()
    # for key in all_regret_dict.keys():
    #     df = pd.DataFrame(dict(Time=np.arange(1, args.T + 1),
    #                            Average_Regret=all_regret_dict[key]))
    #     g = sns.relplot(x="Time", y="Average_Regret", kind="line", data=df, ax=ax)
    # # g.figure.autofmt_xdate()
    plt.show()