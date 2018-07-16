"""
currently deprecated.
"""

import numpy as np
import scipy.io as sio
import os
#import matlab.engine

np.random.seed(10000)


class FeatureExtractor(object):
    GESTURE_NUM = 14
    FINGER_NUM = 2
    SUBJECT_NUM = 20
    FEATURE_NUM = 100
    GLOBAL_FEATURE_NUM = 30#60#30
    HAND_FEATURE_NUM = 100#480#100

    def __init__(self, data_dir, is_full):
        self.data_dir = data_dir
        self.is_full = is_full

    def LoadData(self):
        feature_train = sio.loadmat(os.path.join(self.data_dir, 'feature_hand_global_train.mat'))
        feature_test = sio.loadmat(os.path.join(self.data_dir, 'feature_hand_global_test.mat'))
        x_train = feature_train['feature'][0]
        y_train = feature_train['y_train'][0]
        x_test = feature_test['feature'][0]
        y_test = feature_test['y_test'][0]
        return (x_train, y_train), (x_test, y_test)

    def LoadData2(self):
        feature_train = sio.loadmat(os.path.join(self.data_dir, 'feature_hand_global_train.mat'))
        feature_test = sio.loadmat(os.path.join(self.data_dir, 'feature_hand_global_test.mat'))
        x_train = feature_train['feature_hand'][0]
        x_train2 = feature_train['feature_global'][0]
        y_train = feature_train['y_train'][0]
        x_test = feature_test['feature_hand'][0]
        x_test2 = feature_test['feature_global'][0]
        y_test = feature_test['y_test'][0]
        return (x_train, x_train2, y_train), (x_test, x_test2, y_test)

    def extract_feature(self, test_id, seq_id, M, max_dist_factor, offset1, offset2):
        eng = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
        #eng = matlab.engine.start_matlab()
        cwd = os.getcwd()
        eng.addpath(cwd+'/src/feature/')
        do_pca = 0
        is_normalize = 0
        x_hand, x_global, y = eng.extract_features_for_alldata(self.is_full, test_id, seq_id, do_pca,
                              self.HAND_FEATURE_NUM, M, max_dist_factor, offset1, offset2, is_normalize, nargout=3)
        y = np.array(y[0])
        eng.quit()
        return x_hand, x_global, y

    @property
    def input_dim(self):
        #return self.FEATURE_NUM
        return (self.HAND_FEATURE_NUM, self.GLOBAL_FEATURE_NUM)

    @property
    def class_num(self):
        if self.is_full:
            return self.GESTURE_NUM * self.FINGER_NUM
        else:
            return self.GESTURE_NUM

if __name__ == "__main__":
    fextractor = FeatureExtractor("../data/")
    (x_train, x_train2, y_train), (x_test, x_test2, y_test) = fextractor.LoadData2()
    print y_train.dtype
    print y_train.shape
    print y_train[0].shape
    x_hand, x_global, y = fextractor.extract_feature(1, 0)
    print type(x_hand)
    print type(x_global)
    print type(y)
    print y.shape
    print y.dtype
    print x_hand[0]
