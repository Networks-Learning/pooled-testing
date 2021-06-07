#!/bin/bash

methods=(binomial negbin)
se=0.95
sp=0.95
N_seq=(20 100)
r_seq=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
k_seq=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
seeds=1000000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for v in {0..1}
do
    N=${N_seq[$v]}
    for z in {0..9}
    do
        r=${r_seq[$z]}
        for i in {0..9}
        do
            k=${k_seq[$i]}
            for j in {0..1}
            do 
                python -m src.experiment --output=outputs/score_vs_R_k_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  &
            done
        done
        wait
    done
done