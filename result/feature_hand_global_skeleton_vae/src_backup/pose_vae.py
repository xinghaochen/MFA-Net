'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import gesture_dataset_shrec17
import gesture_dataset
from data_util import show_skeleton, show_two_skeleton

np.random.seed(10000)

is_train = False
is_test = True

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean, z_log_var, z_mean):
        original_dim = 66
        xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_log_var = inputs[2]
        z_mean = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_log_var, z_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
        
class PoseVAE(object):
    def __init__(self, dataset_='DHG2016', test_id_=-1, is_train=0):
        # parameter setting
        self.batch_size = 100
        self.original_dim = 66
        self.latent_dim = 20
        self.intermediate_dim = 32
        self.epochs = 50
        self.epsilon_std = 0.01#0.05
        self.dataset = dataset_
        self.test_id = test_id_
        
        self.save_postfix = 'pose_vae_' + self.dataset.lower()
        self.save_dir = './snapshot/'+self.save_postfix
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if self.test_id >= 0:
            self.snapshot_file = self.save_dir + '/weights_' + self.save_postfix + '_testid_{}'.format(self.test_id) + '.hdf5'
        else:
            self.snapshot_file = self.save_dir + '/weights_' + self.save_postfix + '.hdf5'

        # build model
        self.build_vae()
        if is_train == 0:
            self.build_encoder()
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    

    
    def build_vae(self):
        # build model
        self.x = Input(shape=(self.original_dim,))
        self.h = Dense(self.intermediate_dim, activation='relu')(self.x)
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_var = Dense(self.latent_dim)(self.h)
        
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        
        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        self.h_decoded = self.decoder_h(self.z)
        self.x_decoded_mean = self.decoder_mean(self.h_decoded)
        
        
        self.y = CustomVariationalLayer()([self.x, self.x_decoded_mean, self.z_log_var, self.z_mean])
        self.vae = Model(self.x, self.y)
        self.vae.compile(optimizer='rmsprop', loss=None)
    
    def train(self, x_train, x_test):
        checkpointer = ModelCheckpoint(filepath=self.snapshot_file,
                verbose=1)
            
        self.vae.fit(x_train,
                shuffle=True,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, None),
                callbacks=[checkpointer])
    
    def test(self, x_test):
        # build a model to project inputs on the latent space
        encoder = Model(self.x, self.z_mean)
        encoder.load_weights(self.snapshot_file, by_name=True)
    
        #
        ## display a 2D plot of the digit classes in the latent space
        x_test_encoded = encoder.predict(x_test, batch_size=self.batch_size)
        #plt.figure(figsize=(6, 6))
        #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        #plt.colorbar()
        #plt.show()
        
        # build a generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)
        generator.load_weights(self.snapshot_file, by_name=True)
        print generator.layers[0].get_weights()
        print generator.layers[1].get_weights()
        print generator.layers[2].get_weights()
        
        # display a 2D manifold of the digits
        #n = 15  # figure with 15x15 digits
        #digit_size = 28
        #figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        #grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        #grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        
        for i in xrange(10):#x_test_encoded.shape[0]):
                i = np.random.randint(0, x_test_encoded.shape[0])
                #z_sample = np.random.rand(1, latent_dim)
                #print z_sample.shape
                z = np.reshape(x_test_encoded[i], (1, self.latent_dim))
                x_decoded = generator.predict(z)
                print 'z', z
                pose = x_decoded.reshape(self.original_dim/3, 3)
                pose_gt = x_test[i, :].reshape(self.original_dim/3, 3)
                print 'pose', pose
                print x_decoded.shape
                print 'pose_gt', pose_gt
                show_two_skeleton(pose, pose_gt, is_show_connect=1, is_show_id=0)
        #        figure[i * digit_size: (i + 1) * digit_size,
        #               j * digit_size: (j + 1) * digit_size] = digit
        
        plt.pause(0)
        #plt.figure(figsize=(10, 10))
        #plt.imshow(figure, cmap='Greys_r')
        #plt.show()
        
    def build_encoder(self):
        # build a model to project inputs on the latent space
        self.encoder = Model(self.x, self.z_mean)
        self.encoder.load_weights(self.snapshot_file, by_name=True)
        
    def encode(self, x_test):
        x_test_encoded = self.encoder.predict(x_test, batch_size=self.batch_size)
        return x_test_encoded

if __name__ == "__main__":
    dataset = 'DHG2016'
    for test_id in xrange(1, 21):
        print 'Training PoseVAE for {} dataset and test_id {}'.format(dataset, test_id)
        if dataset == 'DHG2016':
            vae = PoseVAE(dataset, test_id, 1)
            # SHREC17 hand gesture dataset
            root_dir = '/home/workspace/Datasets/DHG2016'
            is_full = 0
            data = gesture_dataset.Dataset(root_dir, is_full)
            (x_train, y_train), (x_test, y_test) = data.load_data(test_id, is_preprocess=False, is_sub_center=True)        
        else:
            vae = PoseVAE(dataset, -1, 1)
            if test_id >0:
                break
            # SHREC17 hand gesture dataset
            root_dir = '/home/workspace/Datasets/HandGestureDataset_SHREC2017'
            is_full = 0
            data = gesture_dataset_shrec17.Dataset(root_dir, is_full)
            (x_train, y_train), (x_test, y_test) = data.load_data(is_preprocess=False, is_sub_center=True)
        
        
        print len(x_train)
        print len(x_test)
        x_train = np.vstack(x_train)
        x_test = np.vstack(x_test)
        print x_train.shape
        print x_test.shape
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        print x_train
        vae.test_id = test_id
        vae.train(x_train, x_test)
        #vae.test(x_test)
