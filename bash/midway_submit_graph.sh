#! /bin/bash



if [ -f job_graph.sh ]
then
		rm job_graph.sh
fi

touch job_graph.sh

tee -a job_graph.sh << EOF
#! /bin/bash

######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/graph.out
#SBATCH --error=./job-outs/graph.err
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

python /home/bincheng/TwoCapital_Bin/abatement/Result_spe_name.py --name "midwaynew"

echo "Program ends \$(date)"

EOF


sbatch job_graph.sh

