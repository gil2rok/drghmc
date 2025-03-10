# Delayed Rejection Generalized Hamiltonian Monte Carlo

## Overview :telescope:

This repository contains experimental code for the paper [Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo](https://arxiv.org/abs/2406.02741), accepted to AIStats 2025.

We propose the delayed rejection generalized Hamiltonian Monte Carlo (DR-G-HMC) algorithm to generate samples from (unnormalized) probability distributions.

DR-G-HMC is a Markov chain Monte Carlo method that locally adapts step sizes in a clever way. This makes DR-G-HMC especially useful for *efficiently* sampling from distributions with *multiscale geometry* (e.g. curvature) that commonly arises in Bayesian hiearchical models.

> [!TIP]
> To use DR-G-HMC, check out the implementation in [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit/blob/main/bayes_kit/drghmc.py), a readable and pedagogical repository for Bayesian inference algorithms in Python.

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

Sampling from a probability distribution is a widespread problem in statistics and machine learning. The goal is to generate samples from a distribution that is difficult to sample from directly. Markov chain Monte Carlo (MCMC) methods are a popular class of algorithms that generate samples from a distribution by constructing a Markov chain that has the desired distribution as its stationary distribution. The chain is then run for a long time to generate samples from the distribution.

Hamiltonian Monte Carlo (HMC) is a popular type of MCMC method that generates samples by simulating the dynamics of a *ficticious* physical system with gradient information. HMC is particularly useful for sampling from high-dimensional distributions but struggles with multiscale geometry, where the curvature of the distribution varies across different regions of the space. DR-G-HMC is a modification of HMC that addresses this issue and improves upon prior work of DR-HMC.

### Papers:
TODO: Add links to papers.

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