#! /bin/bash




action_name="moreiteration"

count=0


for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 

	mkdir -p ./job-outs/${action_name}/${PSI_0}_${PSI_1}/

	if [ -f ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh ]
	then
			rm ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	fi

	mkdir -p ./bash/${action_name}/

	touch ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	
	tee -a ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=test-${count}
#SBATCH --output=./job-outs/${PSI_0}_${PSI_1}/test.out
#SBATCH --error=./job-outs/${PSI_0}_${PSI_1}/test.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

python3 /home/bcheng4/TwoCapital_Bin/abatement/predamage_spe_psi_name.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name $name2

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
