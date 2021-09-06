#!/bin/bash

methods=(binomial negbin)
se=0.8
sp=0.98
d=0.0427
N=50
r_seq=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0)
k_seq=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
seeds=10000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for z in {0..19}
do
    r=${r_seq[$z]}
    for i in {0..19}
    do
        k=${k_seq[$i]}
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/score_vs_R_k_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  &
        done
    done
    wait
done