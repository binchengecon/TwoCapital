#! /bin/bash

res1=$(date +%s.%N)

NUM_DAMAGE=6
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))




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
#SBATCH --job-name=test-job_time
#SBATCH --output=./job-outs/job_time.out
#SBATCH --error=./job-outs/job_time.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"



echo "Program ends \$(date)"
echo "Program ends time ${dd} day ${dh} hour ${dm} minute ${ds} second"

EOF


sbatch job_time.sh

echo "All Done"

# for i in $(seq 0 $((count-1)))
# 		sentence="$(squeue -j ${jobid_post[$i]})"            # read job's slurm status
# 		stringarray=($sentence) 
# 		jobstatus=(${stringarray[12]})            # isolate the status of job number jobid