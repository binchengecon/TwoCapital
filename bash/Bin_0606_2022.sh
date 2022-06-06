#!/bin/bash
#SBATCH --account=pi-lhansen
#SBATCH --mem=64G
#SBATCH --cpus-per-task=25
#SBATCH --partition=standard
#SBATCH --time=2-00:00:00

# Job specific information
#SBATCH --job-name=job1
#SBATCH --output=/home/bcheng4/LogOut/log_postjump.out
#SBATCH --error=/home/bcheng4/LogError/log_postjump.err

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.8/3.8.5

#---------------------------------------------------------------------------------
# Commands to execute below


srun python3 /home/bcheng4/TwoCapital_Bin/tech4D/post_jump_Bin_0606_2022.py
