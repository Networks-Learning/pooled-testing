#!/bin/bash

N_seq=(20 50 100 200)
untraced_seq=(0.0 0.1 0.2 0.5)
methods=(binomial negbin)
se=0.8
sp=0.98
d=0.0455
r=2.5
k=0.1
seeds=10000
njobs=1

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for i in {0..3}
do
    untraced=${untraced_seq[$i]}
    for l in {0..3}
    do
        N=${N_seq[$l]}
        for j in {0..1}
        do 
            python -m src.experiment --output=outputs/hidden_contacts_${methods[$j]}_N_${N}_untraced_${untraced}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --untraced=$untraced --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs & 
            python -m src.experiment --output=outputs/hidden_contacts_bench_${methods[$j]}_N_${N}_untraced_${untraced}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --untraced=$untraced --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs --bench & 
        done
    done
done