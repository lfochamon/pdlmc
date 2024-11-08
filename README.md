# Primal-dual Langevin Monte Carlo (PD-LMC)

Code repository for the paper [Constrained sampling with primal-dual Langevin Monte Carlo](https://arxiv.org/abs/2411.00568).


## Required packages

See [`requirements.txt`](./requirements.txt) for more details:A.
- python
- jax
- numpy
- pandas
- notebook
- matplotlib
- scikit
- tqdm

For the mirror LMC experiments, you also require
- pytorch
- torchvision


## Experiments

- Gaussian sampling with support constraints
    - 1D truncated gaussian
        - Proj. LMC and PD-LMC: [`support_constraint/1d_gaussian.py`](./support_constraint/1d_gaussian.py)
        - Mirror LMC: [`support_constraint/mirror-langevin/1d_gaussian.py`](./support_constraint/mirror-langevin/1d_gaussian.py)
        - PD-LMC with different LMC steps (mini-batch size) [`support_constraint/1d_gaussian_vs_N.py`](./support_constraint/1d_gaussian_vs_N.py)
    - 2D truncated gaussian
        - Proj. LMC and PD-LMC: [`support_constraint/2d_gaussian.py`](./support_constraint/2d_gaussian.py)
        - Mirror LMC: [`support_constraint/mirror-langevin/1d_gaussian.py`](./support_constraint/mirror-langevin/2d_gaussian.py)
        - PD-LMC with different LMC steps (mini-batch size) [`support_constraint/2d_gaussian_vs_N.py`](./support_constraint/2d_gaussian_vs_N.py)
- Bayesian logistic regression with fairness constraints
    - LMC and PD-LMC: [`fairness/fairness.py`](./fairness/fairness.py)
    - PD-LMC for different constraint specifications $\delta$: [`fairness/fairness_relaxed.py`](./fairness/fairness_relaxed.py)
    - PD-LMC with different LMC steps (mini-batch size) [`fairness/fairness_vs_N.py`](./fairness/fairness_vs_N)
- Counterfactual sampling of stock market
    - LMC and PD-LMC (constraining all stocks): [`counterfactual/all_stocks.py`](./counterfactual/all_stocks.py)
    - LMC and PD-LMC (constraining only NVDA and LLY): [`counterfactual/nvda_lly.py`](./counterfactual/nvda_lly.py)

Code to generate all figures and analyzes are available in similarly named Jupyter notebooks.


## Citation

```
@InProceedings{Chamon24c,
    author = "Chamon, L. F. O. and Jaghargh, M. R. K. and Korba, A.",
    title = "Constrained sampling with primal-dual {L}angevin {M}onte {C}arlo",
    booktitle = "Conference on Neural Information Processing Systems (NeurIPS)",
    year = "2024",
}
```


## Acknowledgments

Code for the mirror LMC implementation was based on [this repository](https://github.com/vishwakftw/metropolis-adjusted-MLA).


