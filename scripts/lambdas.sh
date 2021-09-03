#!/bin/bash

N=100
# se=0.9
# sp=0.99
# lambda_1_seq=$(seq 600 700)
# lambda_2_seq=$(seq 0 3 300)
# se=0.8
# sp=0.98
# lambda_1_seq=$(seq 0 300)
# lambda_2_seq=$(seq 0 2 200)
se=0.7
sp=0.97
lambda_1_seq=$(seq 0 100)
lambda_2_seq=$(seq 0 100)
r=2.5
k=0.1
d=0.0427
seeds=100000
njobs=1

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