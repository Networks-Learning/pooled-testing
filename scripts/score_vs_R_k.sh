#!/bin/bash

days=7
methods=(individual dorfman negbin)
lambda_1=0.3
lambda_2=0.2
se=0.7
sp=0.95
r_seq=(0.5 1.0 1.5 2.0 2.5 3.0)
k_seq=(0.1 1.0 10.0 100.0 1000.0 10000.0)
seeds=1000
njobs=45

for z in {0..5}
do
    r=${r_seq[$z]}
    for i in {0..5}
    do
        k=${k_seq[$i]}
        for j in {0..2}
        do 
            python -m src.experiment --output=outputs/score_vs_R_k_${methods[$j]}_days_${days}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --days=$days --r=${r} --k=${k} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs   
        done
    done
done