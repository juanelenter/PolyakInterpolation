#!/bin/bash

SEED=0

for MAPPING in Softmax Taylor
do
    python train.py --seed $SEED --mapping $MAPPING --opt sps --name ${MAPPING}_sps
    python train.py --seed $SEED --mapping $MAPPING --opt sgd --lr 0.01 --name ${MAPPING}_sgd
done



