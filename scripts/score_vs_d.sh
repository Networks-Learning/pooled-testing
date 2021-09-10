#!/bin/bash

methods=(binomial negbin)
N=100
se=0.8
sp=0.98
# d_seq=(0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7 0.725 0.75 0.775 0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975 1.0)
d_seq=$(seq 0.0 0.01 1.0)
r=2.5
k=0.1
seeds=10000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

# for z in {0..40}
for d in $d_seq
do
    # d=${d_seq[$z]}
    for j in {0..1}
    do 
        python -m src.experiment --output=outputs/score_vs_d_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  #&
    done
done