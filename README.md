# Delayed Rejection Generalized Hamiltonian Monte Carlo

This repository contains experimental code for the paper [Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo](https://arxiv.org/abs/2406.02741), accepted to AIStats 2025.

## Overview :telescope:

We propose the delayed rejection generalized Hamiltonian Monte Carlo (**DR-G-HMC**) algorithm to generate samples from an unnormalized probability distribution.

DR-G-HMC is a Markov chain Monte Carlo method that locally adapts step sizes in a clever way. This makes DR-G-HMC especially useful for *efficiently* sampling from distributions with *multiscale geometry*, where the curvature of the target distribution varies across the parameter space. This is a common issue in Bayesian inference for hierarchical models.

To use DR-G-HMC, check out the implementation in [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit/blob/main/bayes_kit/drghmc.py), a readable and pedagogical repository for Bayesian inference algorithms in Python.

## Quick Start :rocket:

Run in your favorite virtual environment:

```bash
# Clone the repository
git clone https://github.com/gil2rok/drghmc
cd drghmc

# Install dependencies
pip install -r requirements.txt
```

## Experiments :scientist:

TODO: Add instructions to run experiments.

## Background + Resources :books:

Generating samples from an unnormalized probability distribution is a key problem in statistics and machine learning. Markov chain Monte Carlo (MCMC) methods are a popular class of algorithms that construct Markov chains with the target distribution as their stationary distribution. The chain is then simulated for a long time to generate samples from the target distribution.

Hamiltonian Monte Carlo (HMC) is an MCMC method that simulates a *ficticious* phyiscal system with gradient information. While effective for high-dimensional distributions, HMC struggles with multiscale geometry. DR-G-HMC builds upon the Delayed Rejection HMC (DR-HMC) algorithm to address this issue.

#### Papers:
- [Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions](https://arxiv.org/abs/2110.00610)
- [Non-reversibly updating a uniform [0,1] value for Metropolis accept/reject decisions](https://arxiv.org/abs/2001.11950)
- [A conceptual introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf)
- [MCMC using Hamiltonian dynamics](https://arxiv.org/pdf/1206.1901.pdf)
- [Tuning-free generalized Hamiltonian Monte Carlo](https://proceedings.mlr.press/v151/hoffman22a/hoffman22a.pdf)
- [Slice sampling](https://arxiv.org/abs/physics/0009028)
- [ATLAS: Adapting Trajectory Lengths and Step-Size for Hamiltonian Monte Carlo](https://arxiv.org/abs/2410.21587)

## Layout :open_file_folder:

```python
drghmc/
├── configs/                # experiment configs
├── data/                   # store sampler draws
├── doc/                    # figures
├── posteriors/             # Stan models with data & reference draws
├── scripts/                # scripts to run experiments
├── src/                    # sampler implementations
```

This repository also includes a `README.md`, `requirements.txt`, `LICENSE`, and `.gitignore` files. The `data` and `posteriors` directories also contain their own `README.md` files with more information. This layout only covers the most important directories.

## Citation :scroll:

```bibtex
@InProccedings{turok2025sampling,
  title={Sampling From Multiscale Densities With Delayed Rejection Generalized Hamiltonian Monte Carlo},
  author={Turok, Gilad and Modi, Chirag and Carpenter, Bob},
  year={2025},
  booktitle={Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research},
}
```