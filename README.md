<div align="center">

# Delayed rejection generalized Hamiltonian Monte Carlo

<!-- ![License](https://img.shields.io/badge/license-MIT-red.svg) 
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Numpy](https://img.shields.io/badge/numpy-1.21.2-blue.svg) -->

</div>
<p align="center">
   [📃 <a href="https://arxiv.org/abs/2406.02741" target="_blank">Paper</a> ] • [📊 Poster WIP] • [🐦 Tweet WIP] • [📄 Citation WIP]<br>
</p>

Code for experiments, manuscript, and poster of [Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo](https://arxiv.org/abs/2406.02741), submitted to NeurIPS 2024.

## Overview :telescope:

We propose the delayed rejection generalized Hamiltonian Monte Carlo (DR-G-HMC) algorithm to simultaneously solve two problems in Markov chain Monte Carlo sampling for Bayesian inference:

- **Multiscale densities**: DR-G-HMC can sample from densities with varying curvature (multiscale) with dynamic step size selection, enabled by making multiple proposal attempts with decreasing step sizes in a single sampling iteration.
- **Inefficient G-HMC**: DR-G-HMC replaces rejections with additional proposal attempts, thereby avoiding inefficient backtracking that plagues the generalized HMC algorithm (G-HMC) upon encountering a rejection.

Extensive emperical experiments demonstrate that DR-G-HMC can indeed (1) sample from multiscale densities and (2) resolve the inefficiencies of G-HMC while (3) maintaining competitiveness with state of the art samplers (NUTS) on non-multiscale densities.

> [!TIP] 
> Readable and stand-alone implementation of the DR-G-HMC sampler can be found in [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit), a repository for Bayesian inference algorithms in Python. Consider giving it a star :star:.

<!-- 
## Details :mag:

**Background:** Markov chain Monte Carlo (MCMC) methods are a class of algorithms to generate samples from intractable probability densities. Gradient-based MCMC methods, such as Hamiltonian Monte Carlo (HMC), are widely successful because of their efficency in high dimensions.

**Problem:** HMC struggles when the target density is *multiscale* i.e. contains curvature that varies throughout the density. In this setting, a large leapfrog step size is needed to *efficiently* explore low curvature regions, while a small leapfrog step size is needed to *accurately* explore high curvature regions.

Multiscale geometry is a pathology that frequently occurs in hiearchical models all over statistics: small changes to top level parameters may induce drastic changes to lower level parameters. (Also note that preconditioning with a mass matrix only helps with *constant* curvature).

**Solution:** We propose a new MCMC sampler, Delayed Rejection Generalized Hamiltonian Monte Carlo (DR-G-HMC), that can efficiently sample from multiscale densities.

With *delayed rejection*, we can make multiple proposal attempts in the same sampling iteration. If a proposal is rejected, we generate a new proposal with a smaller leapfrog step size, and thus larger acceptance probability. If we start with a large initial step size, we can efficiently explore low curvature regions. If the proposal is rejected, we can (repeatedly) generate a new proposal with a smaller step size, and thus larger acceptance probability, in high curvature regions. This allows for *dynamic* step size selection that can sample from multiscale densities.

With *generalized HMC*, we make this approach more efficient. Instead of using a small step size to traverse an entire HMC trajectory (comprised of many leapfrog steps), generalized HMC uses a small step size for a *single* leapfrog step. This allows for step size adaptation only where needed along a trajectory.

**Bonus:**  -->

## Installation :wrench:

```bash
# clone project
git clone https://github.com/gil2rok/drghmc
cd drghmc

# install requirements
pip install -r requirements.txt
```

## How to run :rocket:

some steps

## Code structure :file_folder:

```python
drghmc/
│
├── configs/                # experiment configs
│
├── data/                   # store sampler draws
│
├── doc/                    # figures, paper manuscript, & poster
│
├── drghmc/                 # sampler implementations
│
├── posteriors/             # Stan models with data
│
├── scripts/                # scripts to run experiments
```

This repository also includes a `README.md`, `requirements.txt`, `LICENSE`, `.gitignore`, and `CITATION.bib` files.

## Citation :page_facing_up:

```bibtex
@article{turok2024sampling,
  title={Sampling From Multiscale Densities With Delayed Rejection Generalized Hamiltonian Monte Carlo},
  author={Turok, Gilad and Modi, Chirag and Carpenter, Bob},
  journal={arXiv preprint arXiv:2406.02741},
  year={2024}
}
```

## Resources :books:

#### Relevant Papers
- [Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions](https://arxiv.org/abs/2110.00610)
- [Non-reversibly updating a uniform [0,1] value for Metropolis accept/reject decisions](https://arxiv.org/abs/2001.11950)

#### Background papers
- [A conceptual introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf)
- [MCMC using Hamiltonian dynamics](https://arxiv.org/pdf/1206.1901.pdf)
- [Tuning-free generalized Hamiltonian Monte Carlo](https://proceedings.mlr.press/v151/hoffman22a/hoffman22a.pdf)
- [Slice sampling](https://arxiv.org/abs/physics/0009028)

#### Visualization

Hamiltonian Monte Carlo sampling from a 2-dimensional banana-shaped distribution:

![hmc sampler](https://raw.githubusercontent.com/chi-feng/mcmc-demo/master/docs/hmc.gif)
