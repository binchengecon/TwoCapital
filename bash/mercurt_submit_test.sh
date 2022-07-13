#! /bin/bash

NUM_DAMAGE=6
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))

while getopts 0:1: option;
do
	case $option in
		0) PSI_0=$OPTARG;;
		1) PSI_1=$OPTARG;;
	esac
done

echo "The parameter psi_0 = ${PSI_0}"

echo "The parameter psi_1 = ${PSI_1}"
			

mkdir -p ./job-outs/${PSI_0}_${PSI_1}/


for i in $(seq 0 $ID_MAX_DAMAGE)
do
		if [ -f job-$i.sh ]
		then
				rm job-$i.sh
		fi

		touch job-$i.sh
		
		tee -a job-$i.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=test_$i
#SBATCH --output=./job-outs/${PSI_0}_${PSI_1}/test_$i.out
#SBATCH --error=./job-outs/${PSI_0}_${PSI_1}/test_$i.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module purge
module load python/booth/3.6/3.6.12  gcc/9.2.0


echo "\$SLURM_JOB_NAME"

python3 post_damage.py --xi_a 1000.0 --xi_g 1000.0 --id $i --psi_0 $PSI_0 --psi_1 $PSI_1

echo "Program ends \$(date)"

EOF
done

for i in $(seq 0 $ID_MAX_DAMAGE)
do
		sbatch job-$i.sh
done

