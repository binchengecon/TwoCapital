#! /bin/bash


######## login 
#SBATCH --job-name=post_4
#SBATCH --output=./job-outs/moreiteration/0.010_0.8/mercury_post_1.out
#SBATCH --error=./job-outs/moreiteration/0.010_0.8/mercury_post_1.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"



python3 /home/bcheng4/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --id 1 --psi_0 0.010 --psi_1 0.8 --name moreiteration

echo "Program ends $(date)"

