#!/usr/bin/env bash
DATA_DIR=/home/icvl/xinghao/Datasets/DHG2016

# bid, ml96, save_best_only
for i in `seq 1 20`;
do
    PREFIX=feature_hand_global_skeleton_bid_100e_norm
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 128 -e 100 -s $PREFIX -m 0 -data 4 -bi -nsbo
done
