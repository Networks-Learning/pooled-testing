#!/bin/bash

N=(5 10 20 50 100 200)
methods=(individual dorfman negbin)
lambda_1=0.33
lambda_2=0.33
se=0.7 # https://www.bmj.com/content/369/bmj.m1808
sp=0.95 # https://www.bmj.com/content/369/bmj.m1808
r=0.6 # Ok for now
k=0.1 # Ok for now
seeds=100
njobs=20

for i in {0..5}
do
    for j in {0..2}
    do 
        python -m src.experiment --output=outputs/score_vs_N_${methods[$j]}_N_${N[$i]}_r_${r}_k_${k}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=${N[$i]} --r=$r --k=$k --method=${methods[$j]} --seeds=$seeds --njobs=$njobs   
    done
done
