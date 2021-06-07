#!/bin/bash

N=100
lambda_1_seq=(0.0 100.0 200.0 300.0 400.0 500.0 600.0 700.0 800.0 900.0 1000.0)
lambda_2_seq=(0.0 100.0 200.0 300.0 400.0 500.0 600.0 700.0 800.0 900.0 1000.0)
se_seq=(0.75 0.95)
sp_seq=(0.75 0.95)
# se=0.75
# sp=0.75
r=2.5
k=0.1
seeds=1000000
njobs=1

for z in {0..1}
do
    se=${se_seq[$z]}
    for v in {0..1}
    do
        sp=${sp_seq[$v]}
        for i in {0..10}
        do
            lambda_1=${lambda_1_seq[$i]}
            lambda_2=0
            python -m src.experiment --output=outputs/lambdas_negbin_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=$k --method=negbin --seeds=$seeds --njobs=$njobs &
            lambda_1=0
            lambda_2=${lambda_2_seq[$i]}
            python -m src.experiment --output=outputs/lambdas_negbin_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=$k --method=negbin --seeds=$seeds --njobs=$njobs &
        done
        wait
    done
done
