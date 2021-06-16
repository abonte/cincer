#!/usr/bin/bash

# FASHION MNIST - MNIST
for D in mnist fashion_mnist; do
for L in convnet fullnet; do
for I in margin; do

  files=`ls results/nipsv2/200__${D}__${L}__*I=${I}*`
  python draw.py -o plots --question q2 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q3 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q2 --style-by negotiator --sup ${files}
  python draw.py -o plots --question q3 --style-by negotiator --sup ${files}


done
done
done


for D in mnist; do
for L in logreg; do
for I in margin; do

  files=`ls results/nipsv2/200__${D}__${L}__*I=${I}*`
  python draw.py -o plots --question q2 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q3 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q2 --style-by negotiator --sup ${files}
  python draw.py -o plots --question q3 --style-by negotiator --sup ${files}


done
done
done


# TABULAR DATA SETS

for D in adult breast 20ng; do
for L in fullnet; do
for I in margin; do

  files=`ls results/nipsv2/200__${D}__${L}__*I=${I}*`
  python draw.py -o plots --question q2 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q3 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q2 --style-by negotiator --sup ${files}
  python draw.py -o plots --question q3 --style-by negotiator --sup ${files}


done
done
done
