#! /bin/bash


######## login 
#SBATCH --job-name=test_1
#SBATCH --output=./job-outs/0.012_0.8/test_1.out
#SBATCH --error=./job-outs/0.012_0.8/test_1.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1


python /home/bincheng/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name.py --xi_a 1000.0 --xi_g 1000.0 --id 1 --psi_0 0.012 --psi_1 0.8 --name "midwaynew"

echo "Program ends $(date)"

