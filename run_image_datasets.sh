#!/usr/bin/bash

# --if-damp 0.01 for other than if
shared='--n-repeats 5 --p-known 100 --max-iters 100 --n-epochs 100 --threshold 0.2 --if-damp 1  --lissa-depth 10 --lissa-samples 1 --bit 64'

for L in fullnet convnet; do
for D in mnist fashion_mnist; do
for I in margin; do

# UPPER BOUND
cmd="python -u main.py 200 q3 $D $L  $shared --noise-type random --inspector always -p 0.0"
echo " #### $cmd #### "
$cmd

# NO CE
cmd="python -u main.py 200 q3 $D $L  $shared -p 0.2 --no-ce --noise-type random --inspector $I --negotiator nearest"
echo " #### $cmd #### "
$cmd

for NEG in top_fisher practical_fisher nearest ce_removal; do

cmd="python -u main.py 200 q3 $D $L $shared -p 0.2 --noise-type random --inspector $I --negotiator $NEG"
echo " #### $cmd #### "
$cmd

done
done
done
done


for L in logreg; do
for D in mnist; do
for I in margin; do

# UPPER BOUND
cmd="python -u main.py 200 q3 $D $L  $shared --noise-type random --inspector always -p 0.0"
echo " #### $cmd #### "
$cmd

# NO CE
cmd="python -u main.py 200 q3 $D $L  $shared -p 0.2 --no-ce --noise-type random --inspector $I --negotiator nearest"
echo " #### $cmd #### "
$cmd

for NEG in top_fisher practical_fisher nearest ce_removal; do

cmd="python -u main.py 200 q3 $D $L $shared -p 0.2 --noise-type random --inspector $I --negotiator $NEG"
echo " #### $cmd #### "
$cmd

done
done
done
done