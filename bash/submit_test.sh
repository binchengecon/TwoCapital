#!/bin/bash
#SBATCH --job-name=example_sbatch
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

module load python/anaconda-2020.02
module load gcc/6.1

# python /home/bincheng/hello.py

start=`date +%s`


name="Test"

for psi_1 in 0.8
do 
   for psi_0 in 0.008
   do
        for id in 0
        do
           python /home/bincheng/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name.py --psi_0 $psi_0 --psi_1 $psi_1 --id $id --name $name &
        done
   done
done

wait

# for psi_1 in 0.8
# do
#    for psi_0 in 0.008 0.010 0.012
#    do 
#        python /home/bincheng/TwoCapital_Bin/abatement/predamage_spe_psi_name.py --psi_0 $psi_0 --psi_1 $psi_1 --name $name &
#    done
# done

# wait 

# python /home/bincheng/TwoCapital_Bin/abatement/Result_spe_name.py --name $name

# wait

echo "All Done"

end=`date +%s`

runtime=$((end-start))