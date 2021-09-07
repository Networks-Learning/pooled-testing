#!/bin/bash

methods=(binomial negbin)
se=0.8
sp=0.98
d_seq=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2)
N=50
r=2.5
k_seq=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
seeds=10000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for z in {0..19}
do
    d=${d_seq[$z]}
    for i in {0..19}
    do
        k=${k_seq[$i]}
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/score_vs_k_d_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  &
        done
    done
    wait
done