#!/bin/bash
#SBATCH --job-name=example_sbatch
#SBATCH --output=extreme_parallel.out
#SBATCH --error=extreme_parallel.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=28
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

module load python/anaconda-2020.02
module load gcc/6.1

# python /home/bincheng/hello.py

for psi_1 in 0.8
do 
   for psi_0 in 0.008 0.010 0.012
   do
        for id in 0 1 2 3 4 5 
        do
           python /home/bincheng/TwoCapital_Bin/abatement/abatement_postdamage.py --psi_0 $psi_0 --psi_1 $psi_1 --id $id &
        done
   done
done

wait

for psi_1 in 0.8
do
   for psi_0 in 0.008 0.010 0.012
   do 
       python /home/bincheng/TwoCapital_Bin/abatement/abatement_predamage.py --psi_0 $psi_0 --psi_1 $psi_1 &
   done
done

wait 

python /home/bincheng/TwoCapital_Bin/abatement/Result_para2.py

wait

echo "All Done"
