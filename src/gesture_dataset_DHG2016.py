import os
import sys
import collections
import numpy as np
import operator
from data_util import normalize_sequences
from pyfeature import motion_feature as mf
from keras.preprocessing.sequence import pad_sequences
from numpy.matlib import repmat
import pose_vae

import data_util

import scipy.io as sio

np.random.seed(10000)


class Dataset(object):
    GESTURE_NUM = 14
    FINGER_NUM = 2
    SUBJECT_NUM = 20
    FEATURE_NUM = 66
    GLOBAL_FEATURE_NUM = 30
    HAND_FEATURE_NUM = 100

    def __init__(self, root_dir, is_full):
        self.root_dir = root_dir
        self.is_full = is_full

        # Load list
        self.info = collections.defaultdict(list)
        self.line_id = collections.defaultdict(list)
        with open(os.path.join(root_dir, "informations_troncage_sequences.txt")) as f:
            idx = 0
            for line in f:
                # gesture, finger, subject, essai, begin, end
                items = map(int, line.strip().split())
                self.info[items[2]].append(items)
                self.line_id[items[2]].append(idx)
                idx += 1

    def save_testid_corresponding_lineid(self, is_full=False):
        line_id_mat = []
        for i in xrange(1, self.SUBJECT_NUM + 1):
            for line_id in self.line_id[i]:
                line_id_mat.append([i, line_id])
        if is_full:
            sio.savemat('../data/DHGdata/DHGdata_full_lineid.mat', {'line_id_mat': line_id_mat})
        else:
            sio.savemat('../data/DHGdata/DHGdata_lineid.mat', {'line_id_mat': line_id_mat})
        print 'data/DHGdata/DHGdata_full_lineid.mat' + ' saved!'

    def _get_pose(self, info):
        gesture, finger, subject, essai, begin, end = info
        name = "{0}/gesture_{1}/finger_{2}/subject_{3}/essai_{4}/skeleton_world.txt".format(
                self.root_dir, gesture, finger, subject, essai)
        with open(name) as f:
            result = [map(float, line.strip().split()) for line in f]
        return np.array(result[begin:end + 1])

    def load_data(self, test_id, load_test_only = False, is_preprocess=True, is_sub_center=False):
        assert(test_id >= 1 and test_id <= self.SUBJECT_NUM)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in xrange(1, self.SUBJECT_NUM + 1):
            if load_test_only and i != test_id:
                continue
            print 'loading subject {} ...'.format(i)
            sys.stdout.flush()
            cnt = 0
            for info in self.info[i]:
                cnt += 1
                if cnt%10 == 0:
                    print '\t {}/{} loaded.'.format(cnt, len(self.info[i]))
                    sys.stdout.flush()
                pose = self._get_pose(info)
                label = ((info[0] - 1) * self.FINGER_NUM + info[1] - 1
                         if self.is_full else info[0] - 1)

                if is_sub_center:
                    #print pose_aug.shape
                    pose = pose - repmat(pose[:, 3:6], 1, pose.shape[1]/3)
                    pose = (pose/0.1 + 1)/2
                if i == test_id:
                    x_test.append(pose)
                    y_test.append(label)
                else:
                    x_train.append(pose)
                    y_train.append(label)
        if is_preprocess:
            # TODO: preprocess the pose
            # extract the difference to the beginning frame
            normalize_sequences(x_train)
            normalize_sequences(x_test)
        return (x_train, y_train), (x_test, y_test)

    def load_data_with_vae_feature(self, test_id, M, max_dist_factor, offset1, offset2, is_preprocess=True, load_test_only = False):
        assert(test_id >= 1 and test_id <= self.SUBJECT_NUM)
        x_train_skeleton = []
        x_train_global = []
        x_train_hand = []
        y_train = []
        x_test_skeleton = []
        x_test_global = []
        x_test_hand = []
        y_test = []
        vae = pose_vae.PoseVAE(dataset_='DHG2016', test_id_=test_id)
        for i in xrange(1, self.SUBJECT_NUM + 1):
            if load_test_only and i != test_id:
                continue
            print 'loading subject {} ...'.format(i)
            sys.stdout.flush()
            cnt = 0
            for info in self.info[i]:
                cnt += 1
                if cnt%10 == 0:
                    print '\t {}/{} loaded.'.format(cnt, len(self.info[i]))
                    sys.stdout.flush()
                pose = self._get_pose(info)
                label = ((info[0] - 1) * self.FINGER_NUM + info[1] - 1
                         if self.is_full else info[0] - 1)

                feature_hand, feature_global = \
                    mf.extract_all_feature_for_frame_vae(pose, test_id, M, max_dist_factor, offset1, offset2, vae)
                if i == test_id:
                    x_test_skeleton.append(pose)
                    x_test_global.append(feature_global)
                    x_test_hand.append(feature_hand)
                    y_test.append(label)
                else:
                    x_train_skeleton.append(pose)
                    x_train_global.append(feature_global)
                    x_train_hand.append(feature_hand)
                    y_train.append(label)
        if is_preprocess:
            # TODO: preprocess the pose
            # extract the difference to the beginning frame
            normalize_sequences(x_train_skeleton)
            normalize_sequences(x_test_skeleton)
        x_train = [x_train_hand, x_train_global, x_train_skeleton]
        x_test = [x_test_hand, x_test_global, x_test_skeleton]

        return (x_train, y_train), (x_test, y_test)


    @property
    def input_dim(self):
        return self.HAND_FEATURE_NUM, self.GLOBAL_FEATURE_NUM, self.FEATURE_NUM

    @property
    def class_num(self):
        if self.is_full:
            return self.GESTURE_NUM * self.FINGER_NUM
        else:
            return self.GESTURE_NUM


if __name__ == "__main__":
    data = Dataset("/home/workspace/Datasets/DHG2016", is_full=True)
    data.save_testid_corresponding_lineid(is_full=True)

