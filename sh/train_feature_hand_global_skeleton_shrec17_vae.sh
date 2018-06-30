#!/usr/bin/env bash
#DATA_DIR=/home/workspace/Datasets/HandGestureDataset_SHREC2017
DATA_DIR=/home/workspace/data/handgesture/HandGestureDataset_SHREC2017

# bid, ml96, save_best_only
PREFIX=shrec17_feature_hand_global_skeleton_vae_aug_dropout
TF_CPP_MIN_LOG_LEVEL=1 python -u src/main.py -d $DATA_DIR -dataset SHREC17 -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 7 -ml 100 -f 0 -nsbo
