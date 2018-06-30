#!/usr/bin/env bash
DATA_DIR=/home/ghk/data/DHG2016

# bid, ml96, save_best_only
for i in `seq 1 20`;
do
    PREFIX=feature_hand_global_skeleton_ml96_bid_testid_
    PREFIX="$PREFIX$i"
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 4 -ml 96 -bi 1
done

# bid, ml96, save_best_only, full
for i in `seq 1 20`;
do
    PREFIX=feature_hand_global_skeleton_ml96_bid_full_testid_
    PREFIX="$PREFIX$i"
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 4 -ml 96 -bi 1 -f 1
done

# bid, ml96, save_best_only, skeleton
for i in `seq 1 20`;
do
    PREFIX=skeleton_ml96_bid_testid_
    PREFIX="$PREFIX$i"
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 1 -ml 96 -bi 1
done

# bid, ml96, save_best_only, full, skeleton
for i in `seq 1 20`;
do
    PREFIX=skeleton_ml96_bid_full_testid_
    PREFIX="$PREFIX$i"
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 1 -ml 96 -bi 1 -f 1
done

# bid, ml96, save_best_only, feature_hand_global
for i in `seq 1 20`;
do
    PREFIX=feature_hand_global_ml96_bid_testid_
    PREFIX="$PREFIX$i"
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 3 -ml 96 -bi 1
done

# bid, ml96, save_best_only, full, feature_hand_global
for i in `seq 1 20`;
do
    PREFIX=feature_hand_global_ml96_bid_full_testid_
    PREFIX="$PREFIX$i"
    python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 3 -ml 96 -bi 1 -f 1
done