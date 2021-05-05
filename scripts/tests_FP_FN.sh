#!/bin/bash

methods=(binomial negbin)
lambdas_seq=$(seq 0 10)
N=100
se_seq=(0.75 0.95)
sp_seq=(0.75 0.95)
r_seq=(0.5 2.5)
k_seq=(0.1 0.2)
seeds=100000
njobs=4

for i in {0..1}
do
    r=${r_seq[$i]}
    k=${k_seq[$i]}
    for z in {0..1}
    do
        se=${se_seq[$z]}
        for v in {0..1}
        do
            sp=${sp_seq[$v]}
            for lambdas in $lambdas_seq
            do
                for lambda_1 in $(seq 0 $lambdas)
                do
                    lambda_2=$(echo $lambdas - $lambda_1|bc)
                    lambda_1=$(echo "scale=1 ; $lambda_1 / 10" | bc)
                    lambda_2=$(echo "scale=1 ; $lambda_2 / 10" | bc)
                    for j in {0..1}
                    do 
                        python -m src.experiment --output=outputs/tests_FP_FN_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs   
                    done
                done
            done
        done
    done
done