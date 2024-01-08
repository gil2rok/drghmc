#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/slurm-%j.log
date;hostname;id;pwd

module -q purge
source ~/mambaforge/bin/activate drghmc

experiment='funnel10'
python src/runners/runner.py --experiment $experiment