#!/usr/bin/env bash
DATA_DIR=/home/workspace/Datasets/HandGestureDataset_SHREC2017
#DATA_DIR=/home/workspace/data/handgesture/HandGestureDataset_SHREC2017

# bid, ml96, save_best_only
PREFIX=shrec17_feature_hand_global_skeleton_vae_aug_171_noaug

echo $0

# mkdir
cd result
if [ ! -d $PREFIX ]; then
    mkdir $PREFIX
fi
cd $PREFIX
if [ ! -d src_backup ]; then
    mkdir src_backup
fi
# backup all files
cd ../..
find src -name *.py -exec cp {} result/${PREFIX}/src_backup/ \;
cp $0 result/${PREFIX}/

# log file
base_tmp=result/${PREFIX}/log_
TXT=.txt
LOGFILE=result/${PREFIX}/log_${PREFIX}.txt
touch $LOGFILE
echo $LOGFILE
exec > >(tee $LOGFILE)
exec 2>&1

TF_CPP_MIN_LOG_LEVEL=1 python -u src/main.py -d $DATA_DIR -dataset SHREC17 -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 7 -ml 171 -f 0 -nsbo
