#! /bin/bash

NUM_DAMAGE=6
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))

res1=$(date +%s.%N)





for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in 0.008 0.010 0.012
	do
		for PSI_1 in 0.8
		do 

		mkdir -p ./job-outs/${PSI_0}_${PSI_1}/

		if [ -f job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh ]
		then
				rm job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh
		fi

		touch job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh
		
		tee -a job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=test_$i
#SBATCH --output=./job-outs/${PSI_0}_${PSI_1}/test_$i.out
#SBATCH --error=./job-outs/${PSI_0}_${PSI_1}/test_$i.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1


name2="midwaytryname"
echo "\$SLURM_JOB_NAME"

python /home/bincheng/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name.py --xi_a 1000.0 --xi_g 1000.0 --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name "midwaytryname"

echo "Program ends \$(date)"

EOF
		done
	done
done



for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 

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


name2="midwaytryname"
echo "\$SLURM_JOB_NAME"

python /home/bincheng/TwoCapital_Bin/abatement/predamage_spe_psi_name.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name "midwaytryname"

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
#SBATCH --job-name=test
#SBATCH --output=./job-outs/test.out
#SBATCH --error=./job-outs/test.err
#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1

name2="midwaytryname"
echo "\$SLURM_JOB_NAME"

python /home/bincheng/TwoCapital_Bin/abatement/Result_spe_name.py --name "midwaytryname"

echo "Program ends \$(date)"

EOF














# for i in $(seq 0 $ID_MAX_DAMAGE)
# do
# 	for PSI_0 in 0.008 0.010 0.012
# 	do
# 		for PSI_1 in 0.8
# 		do 
# 		sbatch --wait job_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh &

# 		done
# 	done
# done

# wait

# for PSI_0 in 0.008 0.010 0.012
# do
# 	for PSI_1 in 0.8
# 	do 
# 	sbatch --wait job_PSI0_${PSI_0}_PSI1_${PSI_1}.sh &
# 	done
# done

# wait 

# sbatch job_graph.sh

echo "All Done"


# do stuff in here

res2=$(date +%s.%N)
dt=$(echo "$res2 - $res1" | bc)
dd=$(echo "$dt/86400" | bc)
dt2=$(echo "$dt-86400*$dd" | bc)
dh=$(echo "$dt2/3600" | bc)
dt3=$(echo "$dt2-3600*$dh" | bc)
dm=$(echo "$dt3/60" | bc)
ds=$(echo "$dt3-60*$dm" | bc)


if [ -f job_time.sh ]
then
		rm job_time.sh
fi

touch job_time.sh

tee -a job_time.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=time
#SBATCH --output=./job-outs/time.out
#SBATCH --error=./job-outs/time.err
#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1

name2="midwaytryname"
echo "\$SLURM_JOB_NAME"


echo "Program ends \$(dd) $(dh) $(dm) $(ds)"

EOF

sbatch job_time.sh