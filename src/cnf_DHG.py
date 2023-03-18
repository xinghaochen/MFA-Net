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

import gesture_dataset_DHG2016
import scipy.io as sio

# Set Parameters
mode = '14-labels'

if mode == '14-labels':
  num_classes = 14
else:
  num_classes = 28

data_root = '/home/workspace/data/handgesture/DHG2016/'
out_prefix = 'DHG2016_MFA_Net_vae'
if num_classes == 28:
    out_prefix += '_28'
else:
    out_prefix += '_14'

if mode == '14-labels':
  is_full = 0
  class_name_dict = { 0: 'G', 1: 'T', 2: 'E', 3: 'P',
                      4: 'R-CW', 5: 'R-CCW', 6: 'S-R', 7: 'S-L',
                      8: 'S-U', 9: 'S-D', 10: 'S-X', 11: ' S-V', 12: 'S-+', 13: 'Sh'}
else:
  is_full = 1
  class_name_dict = { 0: 'G(1)', 1:'G(2)', 2: 'T(1)', 3: 'T(2)', 4: 'E(1)', 5:'E(2)', 6: 'P(1)', 7: 'P(2)',
                      8: 'R-CW(1)', 9: 'R-CW(2)', 10: 'R-CCW(1)', 11: 'R-CCW(2)', 12: 'S-R(1)', 13: 'S-R(2)', 14: 'S-L(1)', 15: 'S-L(2)',
                      16: 'S-U(1)', 17: 'S-U(2)', 18: 'S-D(1)', 19: 'S-D(2)', 20: 'S-X(1)', 21: 'S-X(2)', 22: 'S-V(1)', 23: 'S-V(2)',
                      24: 'S-+(1)', 25: 'S-+(2)', 26: 'Sh(1)', 27: 'Sh(2)'}


def computer_confusion_matrix(split):
  # Generate Dataset
  data = gesture_dataset_DHG2016.Dataset(data_root, is_full)
  (x_train, y_train), (x_test, y_test) = data.load_data(split, load_test_only = True, is_preprocess=False, is_sub_center=False)

  y_train = np.array((y_train))
  y_test = np.array((y_test))

  # Compute Confusion Matrix
  outdir = 'result/{}/'.format(out_prefix)
  print outdir+'pred_results_{}_testid_{}.mat'.format(out_prefix, split)
  y_pred = sio.loadmat(outdir+'pred_results_{}_testid_{}.mat'.format(out_prefix, split))['pred_result']
  y_pred = np.argmax(y_pred,axis=1)

  cnf_matrix = confusion_matrix(y_test, y_pred)
  
  return cnf_matrix


if __name__ == "__main__":
  outdir = 'result/{}/'.format(out_prefix)

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
  sns.set(font_scale=1.5*28/num_classes)
  sns.heatmap(df_cm, annot=True, fmt=".2f", cmap=cmap)
  plt.tight_layout(rect=(0.0, 0, 1.12, 1))
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
