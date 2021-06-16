#!/usr/bin/bash

# ========================
# Q2 CE PRECISION AT 5, 10

for L in fullnet; do
for D in adult breast; do
for NEG in top_fisher practical_fisher full_fisher 'if';do

cmd="python -u main.py pr_at_k q3 $D $L --n-repeats 5 --p-known 100 --max-iters 100 \
--n-epochs 100 --p-noise 0.2 --noise-type random --inspector never --negotiator $NEG \
--if-damping 1 --lissa-depth 10 --bits 64 --ce-precision"
echo " #### $cmd #### "
$cmd

done
done
done

for L in fullnet; do
for D in 20ng; do
for NEG in top_fisher practical_fisher 'if';do

cmd="python -u main.py pr_at_k q3 $D $L --n-repeats 5 --p-known 500 --max-iters 100 \
--n-epochs 100 --p-noise 0.2 --noise-type random --inspector never --negotiator $NEG \
--if-damping 1 --lissa-depth 10 --bits 64 --ce-precision"
echo " #### $cmd #### "
$cmd

done
done
done


for L in fullnet convnet; do
for D in mnist fashion_mnist; do
for NEG in top_fisher practical_fisher 'if';do

cmd="python -u main.py pr_at_k q3 $D $L --n-repeats 5 --p-known 100 --max-iters 100 \
--n-epochs 100 --p-noise 0.2 --noise-type random --inspector never --negotiator $NEG \
--if-damping 1 --lissa-depth 10 --bits 64 --ce-precision"
echo " #### $cmd #### "
$cmd

done
done
done

for L in logreg; do
for D in mnist; do
for NEG in top_fisher practical_fisher 'if';do

cmd="python -u main.py pr_at_k q3 $D $L --n-repeats 5 --p-known 100 --max-iters 100 \
--n-epochs 100 --p-noise 0.2 --noise-type random --inspector never --negotiator $NEG \
--if-damping 1 --lissa-depth 10 --bits 64 --ce-precision"
echo " #### $cmd #### "
$cmd

done
done
done

