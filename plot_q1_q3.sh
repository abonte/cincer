#!/usr/bin/env bash

FOLDER=results/nipsv2

draw () {

  files=`ls \
          ${FOLDER}/200__$1__$2__*I=$3* \
          ${FOLDER}/200__$1__$2__*p=0.0*I=always*`
  python draw.py -o plots --question q1 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q3 --style-by negotiator --summary ${files}
  python draw.py -o plots --question q1 --style-by negotiator --sup ${files}
  python draw.py -o plots --question q3 --style-by negotiator --sup ${files}

}

# FASHION MNIST - MNIST
for D in mnist fashion_mnist; do
for L in convnet fullnet; do
for I in margin; do

  draw $D $L $I

done
done
done


for D in mnist; do
for L in logreg; do
for I in margin; do

  draw $D $L $I

done
done
done


# TABULAR DATA SETS

for D in adult breast 20ng; do
for L in fullnet; do
for I in margin; do

  draw $D $L $I

done
done
done
