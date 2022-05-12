# k-experts-problem 


The entire code is written in Python 3.6. This is an ongoing work.

## Requirements
All the requirements are mentioned in `requirements.txt`. <br/>
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

## Execution
### Individual algorithms
* Hedge: `python sage_hedge.py with base_config method='METHOD_NAME' -p`
* OCO: `python sage_oco.py with base_config -p`
* FTPL: `python ftpl.py with base_config -p`

### Variants of k-experts
* Sum-rewards (k-sets): `python main.py with base_config -p`

## Citation
If you find this repo useful in your research, please consider to cite the following paper:

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
