# Delayed Rejection Generalized Hamiltonian Monte Carlo

## Overview :telescope:

This repository contains experimental code for the paper [Sampling From multiscale densities with delayed rejection generalized Hamiltonian Monte Carlo](https://arxiv.org/abs/2406.02741), accepted to AIStats 2025.

We propose the delayed rejection generalized Hamiltonian Monte Carlo (DR-G-HMC) algorithm to generate samples from (unnormalized) probability distributions. DR-G-HMC is a Markov chain Monte Carlo method that locally adapts step sizes in a clever way. This makes DR-G-HMC especially useful for *efficiently* sampling from distributions with *multiscale geometry* (e.g. curvature) that commonly arises in Bayesian hiearchical models.

> [!TIP]
> To use DR-G-HMC, check out the implementation in [Bayes-Kit](https://github.com/flatironinstitute/bayes-kit/blob/main/bayes_kit/drghmc.py), a pedagogical and readable repository for Bayesian inference algorithms in Python. Consider starring it :star:.

## Installation :wrench:

Run in your favorite virtual environment:

```bash
# Clone the repository
git clone https://github.com/gil2rok/drghmc
cd drghmc

# Install dependencies
pip install -r requirements.txt
```

## Quick Start :rocket:

## Experiments :scientist:

## Citation :scroll:

## Layout :open_file_folder:

## Background and Resources :books:


