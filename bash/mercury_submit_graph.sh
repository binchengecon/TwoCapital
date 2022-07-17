#! /bin/bash

action_name="name"

if [ -f ./bash/${action_name}/job_graph.sh ]
then
		rm ./bash/${action_name}/job_graph.sh
fi
mkdir -p ./bash/${action_name}/

touch ./bash/${action_name}/job_graph.sh

tee -a ./bash/${action_name}/job_graph.sh << EOF

#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/${action_name}/graph.out
#SBATCH --error=./job-outs/${action_name}/graph.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=highmem
#SBATCH --mem=80G
#SBATCH --time=10:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "\$SLURM_JOB_NAME"

python3 /home/bcheng4/TwoCapital_Bin/abatement/Result_spe_name_moreiteration.py --name  ${action_name}

echo "Program ends \$(date)"

EOF


sbatch ./bash/${action_name}/job_graph.sh

