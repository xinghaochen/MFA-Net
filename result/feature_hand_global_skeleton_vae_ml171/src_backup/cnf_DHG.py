import seaborn as sns
import pandas as pd
import os
import sys
import math
import numpy as np
import random as rn
import time

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras import backend as K

# import Models
from keras.models import Model
from keras.optimizers import RMSprop,SGD,Adam
from keras.regularizers import l2,l1
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
import tensorflow as tf
from keras import backend as K
import gesture_dataset
import scipy.io as sio

# Set Parameters
tight_flag = 1
mode = '14-labels'

filter_length = 8
kernel_regularizer = l1(1.e-4)
dropout = 0.5
activation = 'relu'
lr = 0.01
epochs = 200
batch_size = 256

if mode == '14-labels':
  num_classes = 14
else:
  num_classes = 28
feat_dim = 66
max_len_tight = 150
max_len_loose = 280
if tight_flag == 1:
  max_len = max_len_tight
else:
  max_len = max_len_loose

data_root = '/home/workspace/Datasets/DHG2016/'
out_prefix = 'feature_hand_global_skeleton_vae_noamp'
if num_classes == 28:
    out_prefix += '_full'

if mode == '14-labels':
  is_full = 0
  class_name_dict = { 0: 'G', 1: 'T', 2: 'E', 3: 'P',
                      4: 'R-CW', 5: 'R-CCW', 6: 'S-R', 7: 'S-L',
                      8: 'S-U', 9: 'S-D', 10: 'S-X', 11: ' S-V', 12: 'S-+', 13: 'Sh'}
  '''
  weightdir_dict = {1: '0.907', 2: '0.786', 3: '0.971', 4: '0.936', 5: '0.914',
                    6: '0.800', 7: '0.871', 8: '0.921', 9: '0.914', 10: '0.914',
                    11: '0.907', 12: '0.836', 13: '0.807', 14: '0.893', 15: '0.914',
                    16: '0.950', 17: '0.843', 18: '0.857', 19: '0.979', 20: '0.921'}
  '''
  weightdir_dict = {1: '0.900', 2: '0.721', 3: '0.950', 4: '0.893', 5: '0.914',
                    6: '0.743', 7: '0.864', 8: '0.921', 9: '0.907', 10: '0.907',
                    11: '0.900', 12: '0.800', 13: '0.793', 14: '0.886', 15: '0.929',
                    16: '0.936', 17: '0.786', 18: '0.836', 19: '0.950', 20: '0.850'}

else:
  is_full = 1
  class_name_dict = { 0: 'G(1)', 1:'G(2)', 2: 'T(1)', 3: 'T(2)', 4: 'E(1)', 5:'E(2)', 6: 'P(1)', 7: 'P(2)',
                      8: 'R-CW(1)', 9: 'R-CW(2)', 10: 'R-CCW(1)', 11: 'R-CCW(2)', 12: 'S-R(1)', 13: 'S-R(2)', 14: 'S-L(1)', 15: 'S-L(2)',
                      16: 'S-U(1)', 17: 'S-U(2)', 18: 'S-D(1)', 19: 'S-D(2)', 20: 'S-X(1)', 21: 'S-X(2)', 22: 'S-V(1)', 23: 'S-V(2)',
                      24: 'S-+(1)', 25: 'S-+(2)', 26: 'Sh(1)', 27: 'Sh(2)'}
  '''
  weightdir_dict = {1: '0.843', 2: '0.721', 3: '0.886', 4: '0.807', 5: '0.893',
                    6: '0.793', 7: '0.850', 8: '0.914', 9: '0.771', 10: '0.857',
                    11: '0.921', 12: '0.857', 13: '0.807', 14: '0.821', 15: '0.921',
                    16: '0.936', 17: '0.821', 18: '0.814', 19: '0.900', 20: '0.857'}
  '''
  weightdir_dict = {1: '0.864', 2: '0.643', 3: '0.893', 4: '0.843', 5: '0.829',
                    6: '0.771', 7: '0.771', 8: '0.886', 9: '0.843', 10: '0.850',
                    11: '0.900', 12: '0.779', 13: '0.821', 14: '0.836', 15: '0.907',
                    16: '0.929', 17: '0.800', 18: '0.786', 19: '0.964', 20: '0.807'}


def computer_confusion_matrix(split):
  # Generate Dataset
  data = gesture_dataset.Dataset(data_root, is_full)
  (x_train, y_train), (x_test, y_test) = data.load_data(split, load_test_only = True, is_preprocess=False, is_sub_center=False)

  y_train = np.array((y_train))
  y_test = np.array((y_test))

  # Compute Confusion Matrix
  outdir = '../result/{}/'.format(out_prefix)
  print outdir+'pred_results_{}_testid_{}.mat'.format(out_prefix, split)
  y_pred = sio.loadmat(outdir+'pred_results_{}_testid_{}.mat'.format(out_prefix, split))['pred_result']
  y_pred = np.argmax(y_pred,axis=1)

  cnf_matrix = confusion_matrix(y_test, y_pred)
  
  return cnf_matrix


if __name__ == "__main__":
  outdir = '../result/{}/'.format(out_prefix)

  cnf_matrix = np.zeros((num_classes, num_classes))
  acc_all = np.zeros((20, num_classes))
  for i in range(20):
    cnf_matrix_test_id = computer_confusion_matrix(split=i+1)
    cnf_matrix += cnf_matrix_test_id
    tmp = cnf_matrix_test_id.astype('float') / cnf_matrix_test_id.sum(axis=1)[:, np.newaxis] * 100
    acc_all[i, :] = np.diag(tmp)
    print (i)

  cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]*100
  np.set_printoptions(precision=2)
  df_cm = pd.DataFrame(cnf_matrix, index = [class_name_dict[i] for i in class_name_dict],
                  columns = [class_name_dict[i] for i in class_name_dict])
  fig = plt.figure(figsize = (32,16))
  cmap = sns.cubehelix_palette(light=0.98, as_cmap=True)
  sns.set(font_scale=28/num_classes)
  sns.heatmap(df_cm, annot=True, fmt=".2f", cmap=cmap)
  fig.tight_layout()
  plt.savefig(outdir+"cnf_matrix_DHG{}.png".format(num_classes))
  plt.savefig(outdir+"cnf_matrix_DHG{}.pdf".format(num_classes))

  print 'accuracy:', np.mean(np.diag(cnf_matrix))

  if mode == '14-labels':
      print acc_all
      fine_class = [0, 2, 3, 4, 5]
      coarse_class = [1, 6, 7, 8, 9, 10, 11, 12, 13]
      all_rate = acc_all
      t = np.mean(all_rate, axis=1)
      print 'both:', np.mean(t), np.max(t), np.min(t), np.std(t)
      fine_rate = acc_all[:, fine_class]
      t = np.mean(fine_rate, axis=1)
      print 'fine accuracy:', np.mean(t), np.max(t), np.min(t), np.std(t)
      coarse_rate = acc_all[:, coarse_class]
      t = np.mean(coarse_rate, axis=1)
      print 'coarse accuracy:', np.mean(t), np.max(t), np.min(t), np.std(t)
