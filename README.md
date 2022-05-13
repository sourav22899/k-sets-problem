# k-experts-problem 

This is the official code repo for [k-experts - Online Policies and Fundamental Limits](https://proceedings.mlr.press/v151/mukhopadhyay22a.html), AISTATS 2022. In this paper, we introduce the k-experts problem - a generalization of the classic Prediction with Expert’s Advice framework. Unlike the classic version, where the learner selects exactly one expert from a pool of N experts at each round, in this problem, the learner selects a subset of k experts at each round (1<= k <= N). The reward obtained by the learner at each round is assumed to be a function of the k selected experts. The primary objective is to design an online learning policy with a small regret. In this pursuit, we propose SAGE (Sampled Hedge) - a framework for designing efficient online learning policies by leveraging statistical sampling techniques. For a wide class of reward functions, we show that SAGE either achieves the first sublinear regret guarantee or improves upon the existing ones. Furthermore, going beyond the notion of regret, we fully characterize the mistake bounds achievable by online learning policies for stable loss functions. We conclude the paper by establishing a tight regret lower bound for a variant of the k-experts problem and carrying out experiments with standard datasets.

## Requirements
The entire code is written in Python 3.6. All the requirements are mentioned in `requirements.txt`. <br/>
They can be installed using `pip install -r requirements.txt`.

## How to Run

### Individual algorithms
* Hedge: `python sage_hedge.py with base_config method='METHOD_NAME' -p`
* OCO: `python sage_oco.py with base_config -p`
* FTPL: `python ftpl.py with base_config -p`

### Variants of k-experts
* Sum-rewards (k-sets): `python sum_rewards.py with base_config dataset='DATASET_NAME' -p`
* Pairwise-rewards: `python pairwise_rewards.py with base_config dataset='DATASET_NAME' -p`
* Max-rewards: `python scratch.py with base_config dataset='DATASET_NAME' sample='top' -p`
* Monotone-rewards: `python scratch.py with base_config dataset='DATASET_NAME' nfiles=1000 -p`

### Camera ready Plots
* Regret for Sum-rewards/Pairwise-rewards: `python getplots_sum_rewards_regret.py`
* Regret for Max-rewards: `python getplots_max_rewards_regret.py`
* Hit Rates for Sum-rewards and Monotone-rewards: `python getplots_hitrates.py`

The log files are saved in `./logs/`, and plots are saved in `./figures/`. Please refer to the paper for the details of the hyperparameters for each experiment.

## Citation
If you find this repo useful in your research, please consider citing the following paper:

```
@InProceedings{pmlr-v151-mukhopadhyay22a,
  title = 	 { k-experts - Online Policies and Fundamental Limits },
  author =       {Mukhopadhyay, Samrat and Sahoo, Sourav and Sinha, Abhishek},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {342--365},
  year = 	 {2022},
  editor = 	 {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 Mar},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v151/mukhopadhyay22a/mukhopadhyay22a.pdf},
  url = 	 {https://proceedings.mlr.press/v151/mukhopadhyay22a.html},
}
```
