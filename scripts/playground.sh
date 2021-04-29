#!/bin/bash

# N=(5 10 20 50 100 200)
N=100
methods=(dorfman negbin)
se=0.7 # https://www.bmj.com/content/369/bmj.m1808
sp=0.95 # https://www.bmj.com/content/369/bmj.m1808
r=2.5
k=(0.2 0.5 1.0 1000.0 1000000.0)
seeds=1000000
njobs=4

# Optimizing Tests
lambda_1=0.0
lambda_2=0.0

for i in {0..4}
do
    for j in {0..1}
    do 
        python -m src.experiment --output=outputs/playground_${methods[$j]}_N_${N}_r_${r}_k_${k[$i]}_se_${se}_sp_${sp}_l1_${lambda_1}_l2_${lambda_2}_seeds_${seeds}  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=${k[$i]} --method=${methods[$j]} --seeds=$seeds --njobs=$njobs   
    done
done