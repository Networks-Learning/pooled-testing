#!/bin/bash

N=100
d_seq=(0.01 0.05 0.1)
se=0.8
sp=0.98
lambda_1_seq=$(seq 0 200)
lambda_2_seq=$(seq 0 200)
r=2.5
k=0.1
seeds=10000
njobs=1
for i in {0..2}
do
    d=${d_seq[$i]}
    for lambda_1 in $lambda_1_seq
    do
        lambda_2=0
        python -m src.experiment --output=outputs/d_vs_lambdas_negbin_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=$r --k=$k --method=negbin --seeds=$seeds --njobs=$njobs #&
    done
    # wait
    for lambda_2 in $lambda_2_seq
    do
        lambda_1=0
        python -m src.experiment --output=outputs/d_vs_lambdas_negbin_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_d_${d}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --d=$d --n=$N --r=$r --k=$k --method=negbin --seeds=$seeds --njobs=$njobs #&
    done
    # wait
done