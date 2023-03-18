import seaborn as sns
import pandas as pd
import os
import sys
import math
import numpy as np
import random as rn
import time
import scipy.io as sio

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import gesture_dataset_shrec17_aug as gesture_dataset_shrec17


import tensorflow as tf
from keras import backend as K
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed=1234)
rn.seed(12345)
tf.set_random_seed(1234)

# Set Parameters
mode = '28-labels'

if mode == '14-labels':
    num_classes = 14
    is_full = 0
else:
    num_classes = 28
    is_full = 1

if mode == '14-labels':
    class_name_dict = { 0: 'G', 1: 'T', 2: 'E', 3: 'P',
                      4: 'R-CW', 5: 'R-CCW', 6: 'S-R', 7: 'S-L',
                      8: 'S-U', 9: 'S-D', 10: 'S-X', 11: ' S-+', 12: 'S-V', 13: 'Sh'}
else:
    class_name_dict = { 0: 'G(1)', 1:'G(2)', 2: 'T(1)', 3: 'T(2)', 4: 'E(1)', 5:'E(2)', 6: 'P(1)', 7: 'P(2)',
                      8: 'R-CW(1)', 9: 'R-CW(2)', 10: 'R-CCW(1)', 11: 'R-CCW(2)', 12: 'S-R(1)', 13: 'S-R(2)', 14: 'S-L(1)', 15: 'S-L(2)',
                      16: 'S-U(1)', 17: 'S-U(2)', 18: 'S-D(1)', 19: 'S-D(2)', 20: 'S-X(1)', 21: 'S-X(2)', 22: 'S-+(1)', 23: 'S-+(2)',
                      24: 'S-V(1)', 25: 'S-V(2)', 26: 'Sh(1)', 27: 'Sh(2)'}


def computer_confusion_matrix():
    # Generate Dataset
    data_root = '/home/workspace/data/handgesture/HandGestureDataset_SHREC2017'
    out_prefix = 'shrec17_MFA_Net_vae_aug'
    if num_classes == 28:
        out_prefix += '_28'
    else:
        out_prefix += '_14'

    data = gesture_dataset_shrec17.Dataset(data_root, is_full, is_aug=False)
    (x_train, y_train), (x_test, y_test) = data.load_data(is_preprocess=False, is_sub_center=False, load_test_only=True)

    y_train = np.array((y_train))
    y_test = np.array((y_test))

    # Compute Confusion Matrix
    outdir = 'result/{}/'.format(out_prefix)
    print outdir+'pred_results_{}_testid_None.mat'.format(out_prefix)
    y_pred = sio.loadmat(outdir+'pred_results_{}_testid_None.mat'.format(out_prefix))['pred_result']
    y_pred = np.argmax(y_pred,axis=1)

    print y_test.max(), y_pred.max()
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]*100
    np.set_printoptions(precision=2)

    df_cm = pd.DataFrame(cnf_matrix, index = [class_name_dict[i] for i in class_name_dict],
                  columns = [class_name_dict[i] for i in class_name_dict])
    fig = plt.figure(figsize = (32,16))
    cmap = sns.cubehelix_palette(light=0.98, as_cmap=True)
    sns.set(font_scale=1.5*28/num_classes)
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap=cmap)
    plt.tight_layout(rect=(0.0, 0, 1.12, 1))
    plt.savefig(outdir+"cnf_matrix_SHREC{}.png".format(num_classes))
    plt.savefig(outdir+"cnf_matrix_SHREC{}.pdf".format(num_classes))

    print 'accuracy:', accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    computer_confusion_matrix()