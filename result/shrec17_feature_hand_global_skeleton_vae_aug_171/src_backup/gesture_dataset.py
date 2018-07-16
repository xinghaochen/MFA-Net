import os
import sys
import collections
import numpy as np
import operator
from data_util import normalize_sequences
from pyfeature import motion_feature as mf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import Iterator

from numpy.matlib import repmat
import pose_vae

import data_util

import threading

np.random.seed(10000)


class Dataset(object):
    GESTURE_NUM = 14
    FINGER_NUM = 2
    SUBJECT_NUM = 20
    FEATURE_NUM = 66
    GLOBAL_FEATURE_NUM = 30#60#30
    HAND_FEATURE_NUM = 100#480#100

    def __init__(self, root_dir, is_full, use_image=False):
        self.root_dir = root_dir
        self.is_full = is_full
        if use_image:
            raise NotImplementedError("loading images has not implemented")

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
        import scipy.io as sio
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

    def load_data(self, test_id, load_test_only = False, is_preprocess=True, is_sub_center=False, offline_aug=True):
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
                for aug_id in xrange(5):
                    if not offline_aug:
                        if aug_id > 0:
                            break
                    if aug_id == 0:
                        pose_aug = pose
                    elif aug_id == 1:
                        pose_aug = data_util.random_scale_sequence(pose)
                    elif aug_id == 2:
                        pose_aug = data_util.random_shift_sequence(pose)
                    elif aug_id == 3:
                        pose_aug = data_util.random_interpolate_sequence(pose)
                    elif aug_id == 4:
                        pose_aug = data_util.random_noise_sequence(pose)

                    if is_sub_center:
                        #print pose_aug.shape
                        pose_aug = pose_aug - repmat(pose_aug[:, 3:6], 1, pose_aug.shape[1]/3)
                        pose_aug = (pose_aug/0.1 + 1)/2
                    if i == test_id:
                        x_test.append(pose_aug)
                        y_test.append(label)
                        break
                    else:
                        x_train.append(pose_aug)
                        y_train.append(label)
        if is_preprocess:
            # TODO: preprocess the pose
            # extract the difference to the beginning frame
            normalize_sequences(x_train)
            normalize_sequences(x_test)
        return (x_train, y_train), (x_test, y_test)

    def load_data_with_vae_feature(self, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize,
                                   is_preprocess=True, offline_aug=True):
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
            print 'loading subject {} ...'.format(i)
            sys.stdout.flush()
            cnt = 0
            for info in self.info[i]:
                cnt += 1
                if cnt%10 == 0:
                    print '\t {}/{} loaded.'.format(cnt, len(self.info[i]))
                    sys.stdout.flush()
                    #break
                pose = self._get_pose(info)
                label = ((info[0] - 1) * self.FINGER_NUM + info[1] - 1
                         if self.is_full else info[0] - 1)
                for aug_id in xrange(5):
                    if not offline_aug:
                        if aug_id > 0:
                            break
                    #print aug_id
                    if aug_id == 0:
                        pose_aug = pose
                    elif aug_id == 1:
                        pose_aug = data_util.random_scale_sequence(pose)
                    elif aug_id == 2:
                        pose_aug = data_util.random_shift_sequence(pose)
                    elif aug_id == 3:
                        pose_aug = data_util.random_interpolate_sequence(pose)
                    elif aug_id == 4:
                        pose_aug = data_util.random_noise_sequence(pose)

                    feature_hand, feature_global = \
                        mf.extract_all_feature_for_frame_vae(pose_aug, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize, vae)
                    if i == test_id:
                        x_test_skeleton.append(pose_aug)
                        x_test_global.append(feature_global)
                        x_test_hand.append(feature_hand)
                        y_test.append(label)
                        break
                    else:
                        x_train_skeleton.append(pose_aug)
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
        #print x_test_hand
        #print x_test_global
        return (x_train, y_train), (x_test, y_test)

    def load_data_with_feature(self, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize, is_preprocess=True):
        assert(test_id >= 1 and test_id <= self.SUBJECT_NUM)
        x_train_skeleton = []
        x_train_global = []
        x_train_hand = []
        y_train = []
        x_test_skeleton = []
        x_test_global = []
        x_test_hand = []
        y_test = []
        for i in xrange(1, self.SUBJECT_NUM + 1):
            print 'loading subject {} ...'.format(i)
            for info in self.info[i]:
                pose = self._get_pose(info)
                label = ((info[0] - 1) * self.FINGER_NUM + info[1] - 1
                         if self.is_full else info[0] - 1)
                feature_hand, feature_global = \
                    mf.extract_all_feature_for_frame(pose, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize)
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
        #print x_test_hand
        #print x_test_global
        return (x_train, y_train), (x_test, y_test)


    def load_data_with_vae_feature_noamp(self, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize,
                                   is_preprocess=True, offline_aug=True):
        assert(test_id >= 1 and test_id <= self.SUBJECT_NUM)
        self.GLOBAL_FEATURE_NUM = 25
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
            print 'loading subject {} ...'.format(i)
            sys.stdout.flush()
            cnt = 0
            for info in self.info[i]:
                cnt += 1
                if cnt%10 == 0:
                    print '\t {}/{} loaded.'.format(cnt, len(self.info[i]))
                    sys.stdout.flush()
                    #break
                pose = self._get_pose(info)
                label = ((info[0] - 1) * self.FINGER_NUM + info[1] - 1
                         if self.is_full else info[0] - 1)
                for aug_id in xrange(5):
                    if not offline_aug:
                        if aug_id > 0:
                            break
                    #print aug_id
                    if aug_id == 0:
                        pose_aug = pose
                    elif aug_id == 1:
                        pose_aug = data_util.random_scale_sequence(pose)
                    elif aug_id == 2:
                        pose_aug = data_util.random_shift_sequence(pose)
                    elif aug_id == 3:
                        pose_aug = data_util.random_interpolate_sequence(pose)
                    elif aug_id == 4:
                        pose_aug = data_util.random_noise_sequence(pose)

                    feature_hand, feature_global = \
                        mf.extract_all_feature_for_frame_vae_noamp(pose_aug, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize, vae)
                    if i == test_id:
                        x_test_skeleton.append(pose_aug)
                        x_test_global.append(feature_global)
                        x_test_hand.append(feature_hand)
                        y_test.append(label)
                        break
                    else:
                        x_train_skeleton.append(pose_aug)
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


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class GestureDataGenerator(object):
    def __init__(self, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize):
        self.test_id = test_id
        self.do_pca = do_pca
        self.pca_dim = pca_dim
        self.M = M
        self.max_dist_factor = max_dist_factor
        self.offset1 = offset1
        self.offset2 = offset2
        self.is_normalize = is_normalize
        self.vae = pose_vae.PoseVAE(dataset_='DHG2016', test_id_=test_id)

    def generate(self, x, y, shuffle=True, batch_size=32):
        """
        generate batch data with random augmentation for training
        :param self: 
        :param x: (x_hand, x_global, x_skeleton)
        :param y: (y_hand, y_global, y_skeleton)
        :param shuffle: 
        :param batch_size: 
        :return: 
        """
        n_sample = x[0].shape[0]
        # print x[0].shape, n_sample
        n_batch = int(np.ceil(n_sample*1.0 / batch_size))
        # print n_batch
        while 1:
            for batch_index in range(n_batch):
                # shuffle data at the beginning
                if batch_index == 0:
                    index_array = np.arange(n_sample)
                    if shuffle:
                        index_array = np.random.permutation(n_sample)
                current_index = batch_index * batch_size
                if n_sample > current_index + batch_size:
                    current_batch_size = batch_size
                else:
                    current_batch_size = n_sample - current_index
                # print 'batch {}, current_batch_size {}'.format(batch_index, current_batch_size)
                current_index_array = index_array[current_index: current_index + current_batch_size]
                current_index_array = current_index_array.flatten()
                x_batch = []
                y_batch = y[current_index_array]
                for c in range(0, len(x)):
                    x_batch.append(x[c][current_index_array])
                yield x_batch, y_batch

    def generate_with_feature_extraction(self, x, y, shuffle=True, batch_size=32, max_len=64):
        """
        generate batch data with random augmentation for training
        :param self: 
        :param x: x_skeleton
        :param y: y_skeleton
        :param shuffle: 
        :param batch_size: 
        :return: 
        """
        n_sample = len(x)
        # print x[0].shape, n_sample
        n_batch = int(np.ceil(n_sample*1.0 / batch_size))
        # print n_batch
        while 1:
            for batch_index in range(n_batch):
                # shuffle data at the beginning
                if batch_index == 0:
                    index_array = np.arange(n_sample)
                    if shuffle:
                        index_array = np.random.permutation(n_sample)
                current_index = batch_index * batch_size
                if n_sample > current_index + batch_size:
                    current_batch_size = batch_size
                else:
                    current_batch_size = n_sample - current_index
                # print 'batch {}, current_batch_size {}, current_index {}'.format(batch_index, current_batch_size, current_index)
                current_index_array = index_array[current_index: current_index + current_batch_size]
                current_index_array = current_index_array.flatten()
                x_batch = []
                # extract features
                x_global = []
                x_hand = []
                for i, j in enumerate(current_index_array):
                    feature_hand, feature_global = mf.extract_all_feature_for_frame(x[j], self.test_id, self.do_pca, self.pca_dim,
                                                                                    self.M, self.max_dist_factor, self.offset1,
                                                                                    self.offset2, self.is_normalize)
                    x_global.append(feature_global)
                    x_hand.append(feature_hand)
                normalize_sequences(x)
                # get batch data
                y_batch = y[current_index_array]
                x_hand = np.asarray(x_hand)
                x_global = np.asarray(x_global)
                x = np.asarray(x)
                x_batch.append(x_hand)
                x_batch.append(x_global)
                x_batch.append(x[current_index_array])
                #print x.shape, x_hand.shape, x_global.shape
                for c in range(0, len(x_batch)):
                    x_batch[c] = pad_sequences(x_batch[c], maxlen=max_len, dtype=np.float32, padding='pre', truncating='pre')
                yield x_batch, y_batch

    @threadsafe_generator
    def generate_with_vae_feature(self, x, y, shuffle=True, batch_size=32, max_len=64):
        """
        generate batch data with random augmentation for training
        :param self: 
        :param x: x_skeleton
        :param y: y_skeleton
        :param shuffle: 
        :param batch_size: 
        :return: 
        """
        n_sample = len(x)
        # print x[0].shape, n_sample
        n_batch = int(np.ceil(n_sample*1.0 / batch_size))
        # print n_batch
        while 1:
            for batch_index in range(n_batch):
                # shuffle data at the beginning
                if batch_index == 0:
                    index_array = np.arange(n_sample)
                    if shuffle:
                        index_array = np.random.permutation(n_sample)
                current_index = batch_index * batch_size
                if n_sample > current_index + batch_size:
                    current_batch_size = batch_size
                else:
                    current_batch_size = n_sample - current_index
                #print 'batch {}, current_batch_size {}, current_index {}'.format(batch_index, current_batch_size, current_index)
                current_index_array = index_array[current_index: current_index + current_batch_size]
                current_index_array = current_index_array.flatten()
                x_batch = []
                # extract features
                x_global = []
                x_hand = []
                #vae = pose_vae.PoseVAE(dataset_='DHG2016', test_id_=self.test_id)
                for i, j in enumerate(current_index_array):
                    # x[j] = data_util.random_scale_sequence(x[j])
                    # x[j] = data_util.random_shift_sequence(x[j])
                    # x[j] = data_util.random_interpolate_sequence(x[j])
                    # x[j] = data_util.random_noise_sequence(x[j])
                    # print x.shape
                    # print x
                    feature_hand, feature_global = \
                        mf.extract_all_feature_for_frame_vae(x[j], self.test_id, self.do_pca, self.pca_dim,
                                                            self.M, self.max_dist_factor, self.offset1,
                                                            self.offset2, self.is_normalize,
                                                            self.vae)
                    # feature_hand, feature_global = mf.extract_all_feature_for_frame(x[j], self.test_id, self.do_pca, self.pca_dim,
                    #                                                                 self.M, self.max_dist_factor, self.offset1,
                    #                                                                 self.offset2, self.is_normalize)
                    x_global.append(feature_global)
                    x_hand.append(feature_hand)
                normalize_sequences(x)
                # get batch data
                y_batch = y[current_index_array]
                x_hand = np.asarray(x_hand)
                x_global = np.asarray(x_global)
                x = np.asarray(x)
                x_batch.append(x_hand)
                x_batch.append(x_global)
                x_batch.append(x[current_index_array])
                #print x.shape, x_hand.shape, x_global.shape
                for c in range(0, len(x_batch)):
                    x_batch[c] = pad_sequences(x_batch[c], maxlen=max_len, dtype=np.float32, padding='pre', truncating='pre')
                yield x_batch, y_batch

    def flow(self, X, y=None, batch_size=32, shuffle=True, max_len=64, seed=None):
        print 'flow'
        return FeatureIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, max_len=max_len, seed=seed)


class FeatureIterator(Iterator):
    def __init__(self, X, y, gesture_data_generator,
                 batch_size=32, shuffle=True, max_len=64, seed=None):
        if y is not None and len(X) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = np.asarray(X)
        # if self.X.ndim != 3:
        #     raise ValueError('Input data in `FeatureIterator` '
        #                      'should have rank 3. You passed an array '
        #                      'with shape', self.X.shape)
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.gesture_data_generator = gesture_data_generator
        self.global_feature_num = 30  # 60#30
        self.hand_feature_num = 100  # 480#100
        self.mex_len = max_len
        print 'len(X) {} '.format(len(X))
        super(FeatureIterator, self).__init__(len(X), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        x_skeleton = []#np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        x_hand = []#np.zeros(tuple([current_batch_size] + list(self.X.shape)[1] + self.hand_feature_num))
        x_global = []#np.zeros(tuple([current_batch_size] + list(self.X.shape)[1] + self.global_feature_num))
        for i, j in enumerate(index_array):
            print i, j
            x = self.X[j]
            # print x.shape
            # x = data_util.random_resample_sequence(x, 3)
            # x = data_util.random_translate_sequence(x, 0.05)
            x = data_util.random_scale_sequence(x)
            x = data_util.random_shift_sequence(x)
            x = data_util.random_interpolate_sequence(x)
            x = data_util.random_noise_sequence(x)
            # print x.shape
            # print x
            feature_hand, feature_global = \
                mf.extract_all_feature_for_frame_vae(x, self.gesture_data_generator.test_id,
                                                self.gesture_data_generator.do_pca, self.gesture_data_generator.pca_dim,
                                                self.gesture_data_generator.M, self.gesture_data_generator.max_dist_factor,
                                                self.gesture_data_generator.offset1,
                                                self.gesture_data_generator.offset2, self.gesture_data_generator.is_normalize,
                                                self.gesture_data_generator.vae)

            # feature_hand, feature_global = mf.extract_all_feature_for_frame(x, self.gesture_data_generator.test_id,
            #                                     self.gesture_data_generator.do_pca, self.gesture_data_generator.pca_dim,
            #                                     self.gesture_data_generator.M, self.gesture_data_generator.max_dist_factor,
            #                                     self.gesture_data_generator.offset1,
            #                                     self.gesture_data_generator.offset2, self.gesture_data_generator.is_normalize)
            x_hand.append(feature_hand)
            x_global.append(feature_global)
            x_skeleton.append(x)
        normalize_sequences(x_skeleton)
        batch_x = []
        batch_x.append(x_hand)
        batch_x.append(x_global)
        batch_x.append(x_skeleton)
        for c in range(0, len(batch_x)):
            batch_x[c] = pad_sequences(batch_x[c], maxlen=self.mex_len, dtype=np.float32, padding='pre', truncating='pre')
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        # print 'return'
        return batch_x, batch_y


    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        #print 'lock: ', self.lock
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


if __name__ == "__main__":
    data = Dataset("/home/workspace/Datasets/DHG2016", is_full=True)
    data.save_testid_corresponding_lineid(is_full=True)
    #data = Dataset("/home/workspace/Datasets/DHG2016", is_full=False)
    # for test_id in xrange(1,21):
    #     (x_train, y_train), (x_test, y_test) = data.load_data(test_id, is_preprocess=False)
    #     import scipy.io as sio
    #     sio.savemat('data/DHGdata/DHGdata_full_testid_{}.mat'.format(test_id), {'x_train': x_train,  'y_train':y_train, 'x_test':x_test, 'y_test':y_test})
    #     print 'data/DHGdata_full_testid_{}.mat'.format(test_id) + ' saved!'
    # print data.class_num
    # print type(x_train[0])
    # print type(y_train)
    # print len(x_train)
    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # eng.addpath(r'./feature/')
    # ret = eng.test(x_train)
    # print ret
