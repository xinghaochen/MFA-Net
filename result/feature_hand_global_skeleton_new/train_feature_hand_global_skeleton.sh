#!/bin/bash

DATA_DIR=/home/workspace/Datasets/DHG2016
#DATA_DIR=/home/workspace/data/handgesture/DHG2016

# bid, ml96, save_best_only
PREFIX=feature_hand_global_skeleton_new

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
#find . -name *.py -exec cp {} result/${save_postfix}/src_backup/ \;
cp $0 result/${PREFIX}/

# log file
base_tmp=result/${PREFIX}/log_
TXT=.txt
#exec > >(tee $LOGFILE)
#exec 2>&1

for i in `seq 1 20`;
do
    # python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 140 -e 100 -s $PREFIX -m 0 -data 4 -bi -nsbo
    #TF_CPP_MIN_LOG_LEVEL=1 python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 64 -e 100 -s $PREFIX -m 0 -data 7 -ml 100 -f 0 -nsbo
    echo $i
    LOGFILE=result/${PREFIX}/log_${PREFIX}_${i}.txt
    touch $LOGFILE
    echo $LOGFILE
    TF_CPP_MIN_LOG_LEVEL=1 python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 0 -data 5 -nsbo  -ml 100 2>&1 | tee $LOGFILE
done
python src/cal_result.py ${PREFIX}
