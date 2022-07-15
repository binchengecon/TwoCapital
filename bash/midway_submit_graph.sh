#! /bin/bash

action_name="moreiteration"

count=0

if [ -f ./bash/${action_name}/job2_graph.sh ]
then
		rm ./bash/${action_name}/job2_graph.sh
fi

mkdir -p ./bash/${action_name}/

touch ./bash/${action_name}/job2_graph.sh

tee -a ./bash/${action_name}/job2_graph.sh << EOF
#! /bin/bash

######## login 
#SBATCH --job-name=graph
#SBATCH --output=./bash/${action_name}/graph.out
#SBATCH --error=./bash/${action_name}/graph.err
#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1

echo "\$SLURM_JOB_NAME"

python /home/bincheng/TwoCapital_Bin/abatement/Result_spe_name.py --name ${action_name}

echo "Program ends \$(date)"

EOF


sbatch ./bash/${action_name}/job2_graph.sh

