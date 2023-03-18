#!/bin/bash
#DATA_DIR=/home/workspace/Datasets/DHG2016
DATA_DIR=/home/workspace/data/handgesture/DHG2016

# bid, ml96, save_best_only
PREFIX=DHG2016_MFA_Net_vae_28

echo $0
# mkdir
cd result
if [ ! -d $PREFIX ]; then
    mkdir $PREFIX
fi
cd ..

# log file
base_tmp=result/${PREFIX}/log_test
TXT=.txt
LOGFILE=result/${PREFIX}/log_test_${PREFIX}.txt
touch $LOGFILE
echo $LOGFILE
exec > >(tee $LOGFILE)
exec 2>&1

for i in `seq 1 20`;
do
    echo '---------- testing subject ' $i '--------------'
    TF_CPP_MIN_LOG_LEVEL=1 python src/main.py -d $DATA_DIR -i $i -l 0.001 -b 32 -e 100 -s $PREFIX -m 1 -ml 171 -f 1
done
echo 'calculating overall accuracy...'
python src/cal_result.py ${PREFIX}
