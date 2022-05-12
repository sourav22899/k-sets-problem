# k-experts-problem 


The entire code is written in Python 3.6. This is an ongoing work.

All the requirements are mentioned in `requirements.txt`. 
They can be installed using `pip install -r requirements.txt`.

Currently five algorithms are considered:
 * `Hedge`
 * `OCO (Entropic Regularizer)`
 * `FTPL`
 * `LRU`
 * `LFU`
 
To choose the algorithm for which results are needed, modify `algo_list` in `base_config` in `config.py`. By default, all the algorithms are chosen and executed.

Currently, two values of alpha (cache to library size) is considered.
To choose the value of alpha for which results are needed, modify `alpha_list` in `base_config` in `config.py`. By default, 0.01 and 0.05 are chosen and executed.

The log files are saved in `./logs/` and plots are saved in `./figures/`.

## Execute
### Individual algorithms
* Hedge: `python sage_hedge.py with base_config method='METHOD_NAME' -p`
* OCO: `python sage_oco.py with base_config -p`
* FTPL: `python ftpl.py with base_config -p`

### Variants of k-experts
* Sum-rewards (k-sets): `python main.py with base_config -p`
