#! /bin/bash







for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 

	mkdir -p ./job-outs/${PSI_0}_${PSI_1}/

	if [ -f job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh ]
	then
			rm job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	fi

	touch job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	
	tee -a job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=test-${PSI_0}-${PSI_1}
#SBATCH --output=./job-outs/${PSI_0}_${PSI_1}/test.out
#SBATCH --error=./job-outs/${PSI_0}_${PSI_1}/test.err
#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1

name2="midwaynew"
echo "\$SLURM_JOB_NAME"

python /home/bincheng/TwoCapital_Bin/abatement/predamage_spe_psi_name.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name "midwaynew"

echo "Program ends \$(date)"

EOF
	done
done



for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 
	sbatch job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	done
done
