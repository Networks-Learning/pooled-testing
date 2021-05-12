#!/bin/bash

methods=(binomial negbin)
N=100
lambda_1_seq=(0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0)
lambda_2_seq=(0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0)
se_seq=(0.75 0.95)
sp_seq=(0.75 0.95)
r=2.5
k=0.1
seeds=100000
njobs=1

for z in {0..1}
do
    se=${se_seq[$z]}
    for v in {0..1}
    do
        sp=${sp_seq[$v]}
        for i in {0..9}
        do
            lambda_1=${lambda_1_seq[$i]}
            lambda_2=0
            for j in {0..1}
            do 
                python -m src.experiment --output=outputs/lambdas_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs &
            done
            lambda_1=0
            lambda_2=${lambda_2_seq[$i]}
            for j in {0..1}
            do 
                python -m src.experiment --output=outputs/lambdas_${methods[$j]}_N_${N}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs &
            done
        done
        wait
    done
done