#!/bin/bash
DATA_DIR=/media/xiaowei/Work1/Xinghao/Datasets/DHG2016
echo $DATA_DIR
python src/main.py -d $DATA_DIR -i 6 -l 0.001 -b 32 -e 100 -s feature_hand_global_skeleton_full_testid_6 -m 1 -data 4 -f 1
#python main.py -d /home/workspace/Datasets/DHG2016 -i 1 -l 0.01 -b 32 -e 300 -s snapshot -m 1
#python main.py -d /home/ghk/data/DHG2016 -i 1 -l 0.01 -b 32 -e 10 -s snapshot -m 1
