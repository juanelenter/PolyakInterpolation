#!/bin/bash

seed=0
for mapping in Softmax Sparsemax
do
    python train.py --seed $seed --mapping $mapping --name mapping_$seed
done