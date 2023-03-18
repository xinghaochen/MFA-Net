import os
import collections
import numpy as np
import operator
from data_util import normalize_sequences
from pyfeature import motion_feature as mf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import Iterator

from numpy.matlib import repmat
import pose_vae

np.random.seed(10000)


'''
[Hand Gesture SHREC 2017 Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/)
'''
class Dataset(object):
    GESTURE_NUM = 14
    FINGER_NUM = 2
    SUBJECT_NUM = 28
    FEATURE_NUM = 66
    GLOBAL_FEATURE_NUM = 30
    HAND_FEATURE_NUM = 100

    def __init__(self, root_dir, is_full, is_aug = False):
        self.root_dir = root_dir
        self.is_full = is_full
        self.aug_num = 1
        if is_aug:
            self.aug_num = 5
        print 'is_aug', is_aug

        # Load list
        self.info = collections.defaultdict(list)
        # training set
        with open(os.path.join(root_dir, "train_gestures.txt")) as f:
            for line in f:
                # gesture, finger, subject, essai, label_14, label_28, size
                items = map(int, line.strip().split())
                self.info[0].append(items)
        # testing set
        with open(os.path.join(root_dir, "test_gestures.txt")) as f:
            for line in f:
                # gesture, finger, subject, essai, label_14, label_28, size
                items = map(int, line.strip().split())
                self.info[1].append(items)

    def _get_pose(self, info):
        gesture, finger, subject, essai, label_14, label_28, size = info
        name = "{0}/gesture_{1}/finger_{2}/subject_{3}/essai_{4}/skeletons_world.txt".format(
                self.root_dir, gesture, finger, subject, essai)
        with open(name) as f:
            result = [map(float, line.strip().split()) for line in f]
        return np.array(result), label_14, label_28

    def load_data(self, is_preprocess=True, is_sub_center=False, load_test_only = False):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        # train:0, test:1
        for i in xrange(2):
            if load_test_only and i == 0:
                continue
            for info in self.info[i]:
                pose, label_14, label_28 = self._get_pose(info)
                label = (label_28 if self.is_full else label_14) - 1
                if is_sub_center:
                    #print pose.shape
                    pose = pose - repmat(pose[:, 3:6], 1, pose.shape[1]/3)
                    pose = (pose/0.1 + 1)/2
                if i == 1:
                    x_test.append(pose)
                    y_test.append(label)
                else:
                    x_train.append(pose)
                    y_train.append(label)
        if is_preprocess:
            # extract the difference to the beginning frame
            normalize_sequences(x_train)
            normalize_sequences(x_test)
        return (x_train, y_train), (x_test, y_test)


    def load_data_with_vae_feature(self, M, max_dist_factor, offset1, offset2, is_preprocess=True, load_test_only = False):
        x_train_skeleton = []
        x_train_global = []
        x_train_hand = []
        y_train = []
        x_test_skeleton = []
        x_test_global = []
        x_test_hand = []
        y_test = []
        vae = pose_vae.PoseVAE(dataset_='SHREC17')
        # train:0, test:1
        for i in xrange(2):
            idx = 0
            if load_test_only and i == 0:
                continue
            for info in self.info[i]:
                if idx % 100 == 0:
                    print 'loading data {} ...'.format(idx)
                pose, label_14, label_28 = self._get_pose(info)
                label = (label_28 if self.is_full else label_14) - 1

                for aug_id in xrange(self.aug_num):
                    if i == 1 and aug_id > 0:
                        break # do not use augmentation for testing
                    if aug_id == 0:
                        pose_aug = pose
                    elif aug_id == 1:
                        ### Scale
                        f = np.random.uniform(0.8, 1.2)
                        pose_aug = pose * f
                    elif aug_id ==2:
                        ### Shift
                        d = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
                        d = np.tile(d, 22)
                        pose_aug = np.zeros((pose.shape[0], pose.shape[1]))
                        for frame in range(0, pose.shape[0]):
                            if pose[frame, :].all() != 0:
                                pose_aug[frame, :] = pose[frame, :] + d
                    elif aug_id == 3:
                        ### TimeInterpolation
                        pose_aug = np.zeros((pose.shape[0], pose.shape[1]))
                        for frame in range(0, pose.shape[0] - 1):
                            if pose[frame + 1, :].all() != 0:
                                r = np.random.uniform(0, 1)
                                M_r = pose[frame + 1, :] - pose[frame, :]
                                pose_aug[frame, :] = pose[frame + 1, :] - M_r * r
                        pose_aug[pose.shape[0] - 1, :] = pose[pose.shape[0] - 1, :]
                    elif aug_id == 4:
                        ### Noise
                        pose_aug = np.zeros((pose.shape[0], pose.shape[1]))
                        joint_range = np.array((range(22)))
                        np.random.shuffle(joint_range)
                        joint_index = joint_range[:4]
                        n_1 = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
                        n_2 = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
                        n_3 = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
                        n_4 = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
                        for frame in range(pose.shape[0]):
                            if pose[frame, :].all() != 0:
                                x = pose[frame, :]
                                x[joint_index[0]:joint_index[0] + 3] += n_1
                                x[joint_index[1]:joint_index[1] + 3] += n_2
                                x[joint_index[2]:joint_index[2] + 3] += n_3
                                x[joint_index[3]:joint_index[3] + 3] += n_4
                                pose_aug[frame, :] = x
                    feature_hand, feature_global = \
                        mf.extract_all_feature_for_frame_vae(pose_aug, 0, M, max_dist_factor, offset1, offset2, vae)
                    if i == 1:
                        x_test_skeleton.append(pose_aug)
                        x_test_global.append(feature_global)
                        x_test_hand.append(feature_hand)
                        y_test.append(label)
                    else:
                        x_train_skeleton.append(pose_aug)
                        x_train_global.append(feature_global)
                        x_train_hand.append(feature_hand)
                        y_train.append(label)
                    idx += 1
                    # print pose_aug
                    # print feature_hand
                    # print feature_global
                    # exit()
        if is_preprocess:
            # extract the difference to the beginning frame
            normalize_sequences(x_train_skeleton)
            normalize_sequences(x_test_skeleton)
        x_train = [x_train_hand, x_train_global, x_train_skeleton]
        x_test = [x_test_hand, x_test_global, x_test_skeleton]
        #print x_test_hand
        #print x_test_global
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
    data = Dataset("/home/workspace/Datasets/HandGestureDataset_SHREC2017", is_full=True)
    (x_train, y_train), (x_test, y_test) = data.load_data(is_preprocess=False)
    #     import scipy.io as sio
    #     sio.savemat('data/DHGdata/DHGdata_full_testid_{}.mat'.format(test_id), {'x_train': x_train,  'y_train':y_train, 'x_test':x_test, 'y_test':y_test})
    #     print 'data/DHGdata_full_testid_{}.mat'.format(test_id) + ' saved!'
    print data.class_num
    print type(x_train[0])
    print x_train[0].shape
    print type(y_train)
    print len(x_train)
    print len(x_test)

    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # eng.addpath(r'./feature/')
    # ret = eng.test(x_train)
    # print ret
