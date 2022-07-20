#! /bin/bash


######## login 
#SBATCH --job-name=post_8
#SBATCH --output=./job-outs/lessgridpoint/0.012_0.8/test_2.out
#SBATCH --error=./job-outs/lessgridpoint/0.012_0.8/test_2.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1



echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python /home/bincheng/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --id 2 --psi_0 0.012 --psi_1 0.8 --name lessgridpoint

echo "Program ends $(date)"
