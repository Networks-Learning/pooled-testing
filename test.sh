#!/bin/bash

lambda_1=0.3
lambda_2=0.3
se=0.95
sp=0.95
N=40
r=2.3
k=5.0
p=0.575
method=negbin
seeds=10000000
njobs=4

python -m generator --output=outputs/test.json  --lambda_1=$lambda_1 --lambda_2=$lambda_2 --se=$se --sp=$sp --n=$N --r=$r --k=$k --p=$p --method=$method --seeds=$seeds --njobs=$njobs   

