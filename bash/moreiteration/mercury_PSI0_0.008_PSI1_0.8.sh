#! /bin/bash


######## login 
#SBATCH --job-name=pre-0
#SBATCH --output=./job-outs/moreiteration/0.008_0.8/mercury_pre_.out
#SBATCH --error=./job-outs/moreiteration/0.008_0.8/mercury_pre_.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

python3 /home/bcheng4/TwoCapital_Bin/abatement/predamage_spe_psi_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 0.008 --psi_1 0.8 --name moreiteration

echo "Program ends $(date)"
