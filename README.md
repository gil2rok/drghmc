<div align="center">

# Delayed rejection generalized Hamiltonian Monte Carlo

<!-- ![License](https://img.shields.io/badge/license-MIT-red.svg) 
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Numpy](https://img.shields.io/badge/numpy-1.21.2-blue.svg) -->

</div>
<p align="center">
   [📃 <a href="https://arxiv.org/abs/2406.02741" target="_blank">Paper</a> ] • [📊 Poster WIP] • [🐦 Tweet WIP] • [📄 Citation WIP]<br>
</p>

Code for experiments, manuscript, and poster of [Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo](https://arxiv.org/abs/2406.02741) submitted to NeurIPS 2024.

> [!TIP] 
> Readable and stand-alone implementation of the DR-G-HMC sampler can be found in [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit), a repository for Bayesian inference algorithms in Python. Consider giving it a star :star:.

## Overview :telescope:

We propose the delayed rejection generalized Hamiltonian Monte Carlo (DR-G-HMC) algorithm to simulatenously solve two problems in Markov chain Monte Carlo sampling for Bayesian inference:

- **Multiscale densities**: DR-G-HMC can sample from densities with varying curvature (multiscale) with dynamic step size selection enabled by making multiple proposal attempts with decreasing step sizes in a single sampling iteration.
- **Inefficient G-HMC**: DR-G-HMC replaces rejections with additional proposal attempts, thereby avoiding inefficient backtracking that plagues the generalized HMC sampler (G-HMC) upon encountering a rejection.

Experiments demonstrate that DR-G-HMC can indeed (1) sample from multiscale densities and (2) resolve the inefficiencies of G-HMC while (3) maintaining competitiveness with state of the art samplers (NUTS) on non-multiscale densities.

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
