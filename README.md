<div align="center">

# Delayed rejection generalized Hamiltonian Monte Carlo
</div>

<div align="center">

<!-- cleanly separate them -->
[[**Paper**](https://arxiv.org/abs/2406.02741) :page_facing_up:] • [[**Poster**](#overview) :bar_chart:] • [[**Tweet**](#overview) :bird:]

<!-- [**Overview** :telescope:](#overview) | [**Quick Start** :rocket:](#quick-start) | [**Installation** :wrench:](#installation) | [**Experiments** :scientist:](#experiments) | [**Repository Structure** :open_file_folder:](#repository-structure) | [**Citation** :scroll:](#citation) | [**Resources** :books:](#resources) -->

</div>

<!-- <div align="center">

[**Paper** :page_facing_up:](https://arxiv.org/abs/2406.02741) | [**Poster** :bar_chart:](#overview) | [**Tweet** :bird:](#overview) | [**Citation** :scroll:](#citation)

</div> -->

<!-- <p align="center">
   [📃 <a href="https://arxiv.org/abs/2406.02741" target="_blank">Paper</a> ] • [📊 Poster WIP] • [🐦 Tweet WIP] • [📄 Citation WIP]<br>
</p> -->

This repository contains code for experiments, manuscript, and poster of [Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo](https://arxiv.org/abs/2406.02741), submitted to NeurIPS 2024.

<!-- 
[**Paper** :page_facing_up:](https://arxiv.org/abs/2406.02741) | [**Poster** :bar_chart:](#overview) | [**Tweet** :bird:](#overview) | [**Citation** :scroll:](#citation) -->


## Overview :telescope:

We propose the delayed rejection generalized Hamiltonian Monte Carlo (DR-G-HMC) algorithm to simultaneously solve two problems in Markov chain Monte Carlo sampling for Bayesian inference:

- **Multiscale densities**: DR-G-HMC can sample from hierarchical models that exhibit varying curvature (ie multiscale) with dynamic step size selection, enabled by making multiple proposal attempts with decreasing step sizes in a single sampling iteration.
- **Inefficient G-HMC**: DR-G-HMC replaces rejections with additional proposal attempts, thereby avoiding inefficient backtracking that plagues the generalized HMC (G-HMC) algorithm.

Extensive emperical experiments demonstrate that DR-G-HMC can indeed (1) sample from multiscale densities (unlike NUTS!) and (2) resolve the inefficiencies of G-HMC while (3) maintaining competitiveness with state of the art samplers on non-multiscale densities.

## Quick Start :rocket:

> [!TIP]
> To use DR-G-HMC for your own hierarchical model with multiscale geometry, use the [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit/blob/main/bayes_kit/drghmc.py) implementation.
>
> Bayes-Kit is a readable, well documented repository for Bayesian inference algorithms in Python. Consider starring the repository :star: along with others ![Bayes-Kit](https://img.shields.io/github/stars/flatironinstitute/bayes-kit?style=social) !

## Installation :wrench:

To run experiments from the paper, run the following commands in your favorite virtual environment:

```bash
# clone project
git clone https://github.com/gil2rok/drghmc
cd drghmc

# install requirements
pip install -r requirements.txt
```

## Experiments :scientist:

Reproduce experiements from the paper by running the following commands:

## Repository Structure :open_file_folder:

```python
drghmc/
├── configs/                # experiment configs
├── data/                   # store sampler draws
├── doc/                    # figures, paper manuscript, & poster
├── posteriors/             # Stan models with data & reference draws
├── scripts/                # scripts to run experiments
├── src/                    # sampler implementations
```

This repository also includes a `README.md`, `requirements.txt`, `LICENSE`, and `.gitignore` files. The `data` and `posteriors` directories also contain their own `README.md` files with more information.

## Citation :scroll:

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