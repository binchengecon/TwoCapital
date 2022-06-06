#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --mem=40G
#SBATCH --cpus-per-task=25
#SBATCH --partition=standard
#SBATCH --time=2-00:00:00

# Job specific information
#SBATCH --job-name=job1
#SBATCH --output=log_postjump.out
#SBATCH --error=log_postjump.err

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.8/3.8.5

#---------------------------------------------------------------------------------
# Commands to execute below


srun python3 /home/bcheng4/TwoCapital/tech4D/post-jump-gamma.py
