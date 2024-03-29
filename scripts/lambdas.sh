#!/bin/bash

N=100
se_seq=(0.7 0.8 0.9 0.99)
sp_seq=(0.97 0.98 0.99 0.99)
lambda_1_seq=$(seq 0 100)
lambda_2_seq=$(seq 0 100)
r=2.5
k=0.1
d=0.0455
seeds=10000
njobs=1
for i in {0..3}
do
    se=${se_seq[$i]}
    sp=${sp_seq[$i]}
    for lambda_1 in $lambda_1_seq
    do
        lambda_2=0
        python -m src.experiment --output=outputs/lambdas_negbin_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=$r --k=$k --method=negbin --seeds=$seeds --njobs=$njobs &
    done
    wait
    for lambda_2 in $lambda_2_seq
    do
        lambda_1=0
        python -m src.experiment --output=outputs/lambdas_negbin_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=$r --k=$k --method=negbin --seeds=$seeds --njobs=$njobs &
    done
    wait
done