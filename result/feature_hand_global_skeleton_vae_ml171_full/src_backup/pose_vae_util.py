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
from data_util import show_skeleton, show_two_skeleton

np.random.seed(10000)

is_train = False
is_test = True

batch_size = 100
original_dim = 66
latent_dim = 20
intermediate_dim = 32
epochs = 50
epsilon_std = 0.05


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

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

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train

save_postfix = 'pose_vae'
save_dir = '../snapshot/'+save_postfix
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
snapshot_file = save_dir + '/weights_' + save_postfix + '.hdf5'
if is_train:
    checkpointer = ModelCheckpoint(filepath=snapshot_file,
            verbose=1)
        
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            callbacks=[checkpointer])

if is_test:
    #vae.load_weights(snapshot_file)
    
    
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder.load_weights(snapshot_file, by_name=True)

    #
    ## display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    #plt.figure(figsize=(6, 6))
    #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    #plt.colorbar()
    #plt.show()
    
    # build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    generator.load_weights(snapshot_file, by_name=True)
    print generator.layers[0].get_weights()
    print generator.layers[1].get_weights()
    print generator.layers[2].get_weights()
    
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    #digit_size = 28
    #figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    for i in xrange(10):#x_test_encoded.shape[0]):
            i = np.random.randint(0, x_test_encoded.shape[0])
            #z_sample = np.random.rand(1, latent_dim)
            #print z_sample.shape
            z = np.reshape(x_test_encoded[i], (1, latent_dim))
            x_decoded = generator.predict(z)
            print 'z', z
            pose = x_decoded.reshape(original_dim/3, 3)
            pose_gt = x_test[i, :].reshape(original_dim/3, 3)
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
