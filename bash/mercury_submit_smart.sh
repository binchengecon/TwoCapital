#! /bin/bash

NUM_DAMAGE=6
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))






for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in 0.008 0.010 0.012
	do
		for PSI_1 in 0.8
		do 

		mkdir -p ./job-outs/${action_name}/

		if [ -f ./bash/${action_name}/job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh ]
		then
				./bash/${action_name}/rm job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh
		fi

		touch ./bash/${action_name}/job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh
		
		tee -a ./bash/${action_name}/job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=test_$i-${PSI_0}-${PSI_1}
#SBATCH --output=./job-outs/${PSI_0}_${PSI_1}/test_$i.out
#SBATCH --error=./job-outs/${PSI_0}_${PSI_1}/test_$i.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

name2="mercurytryname"
echo "\$SLURM_JOB_NAME"

python3 /home/bcheng4/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name.py --xi_a 1000.0 --xi_g 1000.0 --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name $name2

echo "Program ends \$(date)"

EOF
		done
	done
done



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
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

name2="mercurynew"
echo "\$SLURM_JOB_NAME"

python3 /home/bcheng4/TwoCapital_Bin/abatement/predamage_spe_psi_name.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name $name2
# python3 /home/bcheng4/TwoCapital_Bin/abatement/Result_spe_name.py --name $name2

echo "Program ends \$(date)"

EOF
	done
done



if [ -f job_graph.sh ]
then
		rm job_graph.sh
fi

touch job_graph.sh

tee -a job_graph.sh << EOF

#! /bin/bash


######## login 
#SBATCH --job-name=test-graph
#SBATCH --output=./job-outs/graph.out
#SBATCH --error=./job-outs/graph.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

name2="mercurynew"
echo "\$SLURM_JOB_NAME"

python3 /home/bcheng4/TwoCapital_Bin/abatement/Result_spe_name.py --name $name2

echo "Program ends \$(date)"

EOF














for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in 0.008 0.010 0.012
	do
		for PSI_1 in 0.8
		do 
		sbatch --wait job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh &

		done
	done
done

wait

for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 
	sbatch --wait job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh &
	done
done

wait 

sbatch job_graph.sh

echo "All Done"

# for i in $(seq 0 $((count-1)))
# 		sentence="$(squeue -j ${jobid_post[$i]})"            # read job's slurm status
# 		stringarray=($sentence) 
# 		jobstatus=(${stringarray[12]})            # isolate the status of job number jobid