#!/bin/bash

methods=(binomial negbin)
se=0.8
sp=0.98
N=50
# d_seq=$(seq 0.0 0.01 1.0)
d_seq=$(seq 0.00 0.05 1.0)
r_seq=$(seq 0.25 0.05 5.0)
k=0.1
seeds=20000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for r in $r_seq
do
    for d in $d_seq
    do
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/score_vs_d_R_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs  &
        done
    done
    wait
done