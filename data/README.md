# Data Directory

This directory contains all the data used in the project, organized by posterior, as well as the log files for SLURM errors and outputs.

Each posterior -- consisting of a probabilistic (Stan) model and data -- has its own directory. Every posterior contains many samplers -- e.g. NUTS, DRHMC, DRGHMC -- with different parameters. Each sampler is run for many chains. Each chain has a csv file of draws and a json file containing summary statistics.

The summary statistics include mean of the parameters, mean of the parameters squared, and effective sample size.

For a given posterior with three samplers and two chains, the directory structure looks like this:

```python
posterior
├── 'sampler_01'
│   ├── chain_01
│   │   ├── draws.csv
│   │   └── summary_stats.json
│   ├── chain_02
│       ├── draws.csv
│       └── summary.json
│
├── 'sampler-02'
│   ├── chain_01
│   │   ├── draws.csv
│   │   └── summary_stats.json
│   ├── chain_02
│       ├── draws.csv
│       └── summary_stats.json
│
└── 'sampler_03'
    ├── chain_01
    │   ├── draws.csv
    │   └── summary_stats.json
    ├── chain_02
        ├── draws.csv
        └── summary_stats.json
```