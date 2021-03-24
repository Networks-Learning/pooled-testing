#!/bin/bash

days=7
methods=(individual dorfman negbin)
lambdas_seq=$(seq 0 10)
se=0.7
sp=0.95
r=2.5
k=0.2
seeds=1000
njobs=40

for lambdas in $lambdas_seq
do
    for lambda_1 in $(seq 0 $lambdas)
    do
        lambda_2=$(echo $lambdas - $lambda_1|bc)
        lambda_1=$(echo "scale=1 ; $lambda_1 / 10" | bc)
        lambda_2=$(echo "scale=1 ; $lambda_2 / 10" | bc)
        for j in {0..2}
        do 
            python -m src.experiment --output=outputs/tests_FP_FN_${methods[$j]}_days_${days}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --days=$days --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs   
        done
    done
done