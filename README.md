<div align="center">

# Delayed rejection generalized Hamiltonian Monte Carlo
</div>

<p align="center">
   [📃 <a href="https://arxiv.org/abs/2406.02741" target="_blank">Paper</a> ]• [📊 Poster WIP] • [🐦 Tweet WIP] <br>
</p>

Code for the paper "Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo" submitted to NeurIPS 2024.

> [!TIP] 
> DR-G-HMC is implemented in [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit), a repository for Bayesian inference algorithms in Python. Consider giving it a star :star:.

## Overview :mag_right:

This repository contains the Numpy implementation of the Delayed Rejection Generalized Hamiltonian Monte Carlo (DR-G-HMC) sampler. 

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

## Citation :page_facing_up:

some citation

## Resources :books:

#### Papers
- [Delayed rejection Hamiltonian Monte Carlo for sampling multiscale distributions](https://arxiv.org/abs/2110.00610)
- [Non-reversibly updating a uniform $[0,1]$ value for Metropolis accept/reject decisions](https://arxiv.org/abs/2001.11950)

#### Background Papers
- [A conceptual introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf)
- [MCMC using Hamiltonian dynamics](https://arxiv.org/pdf/1206.1901.pdf)
- [Tuning-free generalized Hamiltonian Monte Carlo](https://proceedings.mlr.press/v151/hoffman22a/hoffman22a.pdf)
- [Slice sampling](https://arxiv.org/abs/physics/0009028)

#### Visualization

Hamiltonian Monte Carlo sampling from a 2-dimensional banana-shaped distribution:

![hmc sampler](https://raw.githubusercontent.com/chi-feng/mcmc-demo/master/docs/hmc.gif)
