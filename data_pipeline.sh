#!/bin/bash
#SBATCH --job-name=eight-schools                # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks aka processes aka chains
#SBATCH --output=data/logs/slurm_%A_%a.out      # Standard output file (%A is the job ID, %a is the task ID)
#SBATCH --error=data/logs/slurm_%A_%a.err       # Standard error file (%A is the job ID, %a is the task ID)

module -q purge
module load slurm openmpi4
source ~/mambaforge/bin/activate drghmc
mpirun -np $SLURM_NTASKS python -m src.data.generate_samples --model_num eight_schools-eight_schools_centered