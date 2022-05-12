from pathlib import Path


def base_config():
    N = 20
    alpha = 0.01
    alpha_list = [0.1 * x for x in range(1, 11)]
    # alpha_list = [0.02, 0.05, 0.1, 0.2, 0.5]
    nfiles = 500
    k = 5
    T = 10
    T_0 = 1000  #
    eta = 0.2
    theta = None
    squeeze_factor = 1
    R = None
    weights = None
    a = None
    W = None
    p = None
    f = None
    f_list = []
    n_iters = 100
    cnt = 0
    algo = "hedge"
    # algo_list = ["hedge", "oco"]
    algo_list = ["hedge", "lru", "ftpl", "lfu"]
    assert all(algo in ["hedge", "oco", "ftpl", "lru", "lfu", "bwa"] for algo in algo_list)
    method = "large"
    assert method in ["iterative", "direct", "large"]
    cache = {}
    fig_root = Path('./figures/')
    log_root = Path('./logs/')
    expt_name = ''
    resume = False
    filename = ''
    sample = "top"
    assert sample in ["top", "random", "bottom", "inv_prop"]
    dataset = "wiki"
    assert dataset in ["wiki", "movielens", "mit", "zipf"]
    lim1 = 1
    lim2 = lim1 + 20
    lp = 3

named_configs = [base_config]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
