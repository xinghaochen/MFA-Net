#!/usr/bin/env bash
DATA_DIR=/home/workspace/Datasets/DHG2016

# bid, ml96, save_best_only
for i in `seq 1 2`;
do
    PREFIX=feature_hand_global_skeleton_bid_100e_noise_4lstm
    #python src/main.py -d $DATA_DIR -i $i -l 0.0005 -b 512 -e 200 -s $PREFIX -m 0 -data 4 -bi -nsbo
    TF_CPP_MIN_LOG_LEVEL=1 python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 133 -e 200 -s $PREFIX -m 0 -data 5 -bi #-nsbo
done
