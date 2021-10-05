# k-sets-problem

The entire code is written in Python 3.6. 
The library size `N=2430` and time steps `T=100863`.

All the requirements are mentioned in `requirements.txt`. They can be installed using `pip install -r requirements.txt`.

Currently six algorithms are considered:
 * `Hedge`
 * `OCO (Entropic Regularizer)`
 * `FTPL`
 * `LRU`
 * `LFU`
 * `Blackwell Approachability (BWA)`
 
To choose the algorithm for which results are needed, modify `algo_list` in `base_config` in `config.py`. By default, all the algorithms are chosen and executed.

Currently, two values of alpha (cache to library size) is considered.
To choose the value of alpha for which results are needed, modify `alpha_list` in `base_config` in `config.py`. By default, 0.01 and 0.05 are chosen and executed.

The log files are saved in `./logs/` and plots are saved in `./figures/`.

## Run
To run: `python main.py with base_config -p`
