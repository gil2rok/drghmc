#!/bin/bash
#SBATCH --job-name=neals_funnel_10dim   # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=40                     # Number of tasks aka processes
#SBATCH --output=logs/slurm_%A_%a.out   # Standard output file (%A is the job ID, %a is the task ID)
#SBATCH --error=logs/slurm_%A_%a.err    # Standard error file (%A is the job ID, %a is the task ID)

module -q purge
module load slurm openmpi4
source ~/mambaforge/bin/activate drghmc
mpirun -np $SLURM_NTASKS python src/main.py --model_num 3