#!/usr/bin/bash

# FASHION MNIST - MNIST
for D in fashion_mnist mnist; do
for L in fullnet convnet; do

  files=`ls results/nipsv2/pr_at_k/pr_at_k__${D}__${L}*`
  python draw.py -o plots --question eval_ce ${files}

done
done


for D in mnist; do
for L in logreg; do

  files=`ls results/nipsv2/pr_at_k/pr_at_k__${D}__${L}*`
  python draw.py -o plots --question eval_ce ${files}

done
done


# NON MNIST

for D in adult breast 20ng; do
for L in fullnet; do

  files=`ls results/nipsv2/pr_at_k/pr_at_k__${D}__${L}*`
  python draw.py -o plots --question eval_ce ${files}

done
done
