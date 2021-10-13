import seaborn as sns
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
    args.N = 2498
    args.dataset = "wiki"

    # hit rates
    data = pd.read_csv(f'./{args.dataset}_{args.N}_hit_rates.csv')
    sns.set_style("darkgrid")
    # sns.set_context("poster")
    # sns.color_palette("gist_heat")
    sns.color_palette("bright")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})

    # sns.set_style('whitegrid')
    # sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.5})

    import pdb;pdb.set_trace()
    data = data.set_index('alphas')
    data.columns = [x.upper() for x in data.columns]
    data = data.rename(columns={"OCO": "SAGE (FTRL)",
                         "HEDGE": "SAGE (Hedge)",
                         "FTPL": "Bhattacharjee et al."}, errors="ignore")

    f, ax = plt.subplots()
    g_results = sns.lineplot(data=data, markers=True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    g_results.set_xlabel("k/N ratio", fontsize=15)
    g_results.set_ylabel("Hit Rate", fontsize=15)
    plt.grid(linestyle='dashed', which='both')
    plt.savefig(f"{args.dataset}_n={args.N}.svg", format="svg")
    plt.savefig(f"{args.dataset}_n={args.N}.pdf", format="pdf")
    plt.show()