#!/bin/bash

days=7
methods=(individual dorfman negbin)
lambda_1=0.33
lambda_2=0.33
se=0.7
sp=0.95
r_seq=$(seq 0.5 0.25 3)
k_seq=$(seq 0.1 0.1 1.0)
seeds=500
njobs=20

for r in $r_seq
do
    for k in $k_seq
    do
        for j in {0..2}
        do 
            python -m src.experiment --output=outputs/score_vs_R_k_${methods[$j]}_days_${days}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --days=$days --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs   
        done
    done
done