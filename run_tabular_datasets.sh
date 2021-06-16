#!/usr/bin/bash

for L in fullnet; do
for D in adult breast 20ng; do
for I in margin; do

# --if-damp 0.01 for other than if
shared='--n-repeats 5  --max-iters 100 --n-epochs 100 --threshold 0.2 --if-damp 1  --lissa-depth 10 --lissa-samples 1 --bit 64'


if [ $D = '20ng' ]; then
	shared="$shared --p-known 500"
else
	shared="$shared --p-known 100"
fi

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
