#! /bin/bash



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


sbatch job_graph.sh

