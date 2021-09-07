#!/bin/bash

methods=(binomial negbin)
se_seq=(0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9)
sp=0.98
d_seq=(0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2)
N=50
r=2.5
k=0.1
seeds=10000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for z in {0..20}
do
    d=${d_seq[$z]}
    for i in {0..20}
    do
        se=${se_seq[$i]}
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/score_vs_k_d_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  #&
        done
    done
    # wait
done