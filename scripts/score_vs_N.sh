#!/bin/bash

N=(20 50 100 200)
methods=(binomial negbin)
se_seq=(0.7 0.8 0.9)
sp_seq=(0.97 0.98 0.99)
d=0.0427
r=2.5
k=0.1
seeds=10000
njobs=5

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0
for l in {0..2}
do
    se=${se_seq[$l]}
    sp=${sp_seq[$l]}
    for i in {0..3}
    do
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/score_vs_N_${methods[$j]}_N_${N[$i]}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=${N[$i]} --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs & 
        done
    done
done