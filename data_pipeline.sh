#!/bin/bash
#SBATCH --job-name=stochastic_volatility        # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=100                            # Number of tasks aka processes aka chains/runs
#SBATCH --output=logs/slurm_%A_%a.out           # Standard output file (%A is the job ID, %a is the task ID)
#SBATCH --error=logs/slurm_%A_%a.err            # Standard error file (%A is the job ID, %a is the task ID)

module -q purge
module load slurm openmpi4
source ~/mambaforge/bin/activate drghmc
mpirun -np $SLURM_NTASKS python -m src.generate_samples --posterior funnel10
python -m src.process_samples --posterior funnel10