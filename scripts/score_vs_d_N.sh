#!/bin/bash

methods=(binomial negbin)
N_seq=(20 50 100 200)
se=0.8
sp=0.98
d_seq=(0.0 0.01 0.02 0.05 0.1 0.2)
r=2.5
k=0.1
seeds=10000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for z in {0..5}
do
    d=${d_seq[$z]}
    for i in {0..3}
    do
        N=${N_seq[$i]}
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/score_vs_d_N_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  &
        done
    done
done