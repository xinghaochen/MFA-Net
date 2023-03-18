#!/usr/bin/env bash
#DATA_DIR=/home/workspace/Datasets/HandGestureDataset_SHREC2017
DATA_DIR=/home/workspace/data/handgesture/HandGestureDataset_SHREC2017

# bid, ml96, save_best_only
PREFIX=shrec17_MFA_Net_vae_aug_28

echo $0

# log file
base_tmp=result/${PREFIX}/log_test_
TXT=.txt
LOGFILE=result/${PREFIX}/log_test_${PREFIX}.txt
touch $LOGFILE
echo $LOGFILE
exec > >(tee $LOGFILE)
exec 2>&1

TF_CPP_MIN_LOG_LEVEL=1 python -u src/main.py -d $DATA_DIR -dataset SHREC17 -l 0.001 -b 32 -e 100 -s $PREFIX -m 1 -ml 171 -f 1
