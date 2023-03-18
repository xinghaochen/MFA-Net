from keras.models import Sequential, Model
from keras.layers.core import Dropout, Activation, Dense, Lambda, Permute, Reshape, RepeatVector
from keras.layers import concatenate, multiply
from keras.layers import LSTM, Input, SpatialDropout1D
#from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import scipy.io as sio
from keras.utils import multi_gpu_model

import tensorflow as tf

np.random.seed(10000)

def get_model_three_branches_new(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
    upper_input = Input(shape=(None, input_dim1), dtype='float32', name='upper_input')
    lower_input = Input(shape=(None, input_dim2), dtype='float32', name='lower_input')
    middle_input = Input(shape=(None, input_dim3), dtype='float32', name='middle_input')

    # Upper RNN layers
    upper_branch = upper_input
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx == len(rnn_cells1)-1:
            upper_branch = cell_type(cell_num, kernel_initializer='he_normal', recurrent_dropout=0.25)(upper_branch)
        elif idx:
            upper_branch = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(upper_branch)
        else:
            upper_branch = cell_type(cell_num, return_sequences=True, input_dim=input_dim1, kernel_initializer='he_normal', recurrent_dropout=0.25)(upper_branch)
        if is_bid:
            upper_branch = Bidirectional(upper_branch) if idx else Bidirectional(upper_branch, input_shape=(None, input_dim1))
        # upper_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            upper_branch = Dropout(dropout_rate)(upper_branch)
    upper_branch = Dense(256, activation='relu')(upper_branch)
    # upper_branch.add(Dense(512, init='normal'))
    upper_branch = BatchNormalization()(upper_branch)
    # upper_branch.add(Activation('relu'))
    upper_branch = Dropout(0.5)(upper_branch)

    # Lower RNN layers
    lower_branch = lower_input
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx == len(rnn_cells2)-1:
            lower_branch = cell_type(cell_num, kernel_initializer='he_normal', recurrent_dropout=0.25)(lower_branch)
        elif idx:
            lower_branch = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(lower_branch)
        else:
            lower_branch = cell_type(cell_num, return_sequences=True, input_dim=input_dim2, kernel_initializer='he_normal', recurrent_dropout=0.25)(lower_branch)
        if is_bid:
            lower_branch = Bidirectional(lower_branch) if idx else Bidirectional(lower_branch, input_shape=(None, input_dim2))
        # lower_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            lower_branch = Dropout(dropout_rate)(lower_branch)
    lower_branch = Dense(256, activation='relu')(lower_branch)
    # upper_branch.add(Dense(512, init='normal'))
    lower_branch = BatchNormalization()(lower_branch)
    # upper_branch.add(Activation('relu'))
    lower_branch = Dropout(0.5)(lower_branch)

    # Middle RNN layers
    middle_branch = middle_input
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx == len(rnn_cells3)-1:
            middle_branch = cell_type(cell_num, kernel_initializer='he_normal', recurrent_dropout=0.25)(middle_branch)
        elif idx:
            middle_branch = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(middle_branch)
        else:
            middle_branch = cell_type(cell_num, return_sequences=True, input_dim=input_dim3, kernel_initializer='he_normal', recurrent_dropout=0.25)(middle_branch)
        if is_bid:
            middle_branch = Bidirectional(middle_branch) if idx else Bidirectional(middle_branch, input_shape=(None, input_dim3))
        # middle_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            middle_branch = Dropout(dropout_rate)(middle_branch)
    middle_branch = Dense(256, activation='relu')(middle_branch)
    # upper_branch.add(Dense(512, init='normal'))
    middle_branch = BatchNormalization()(middle_branch)
    # upper_branch.add(Activation('relu'))
    middle_branch = Dropout(0.5)(middle_branch)

    # Merge
    merged = concatenate([upper_branch, lower_branch, middle_branch])

    # FC layers
    # for fc_dim, dropout_rate in fc_dims:
    fc = merged
    for idx, fc_cell in enumerate(fc_dims):
        fc_dim, dropout_rate = fc_cell
        fc = Dense(fc_dim, activation='relu')(fc)
        if dropout_rate > 0:
            fc = Dropout(dropout_rate)(fc)
    fc = Dense(output_dim)(fc)
    fc = Activation('softmax')(fc)

    model =  Model(inputs=[upper_input, lower_input, middle_input], outputs=fc)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def train_model(model, x_train, y_train, x_test, y_test, class_num, batch_size,
                num_epoch, save_postfix, max_len, test_id):
    if len(x_train) != 3:
        raise ValueError('Expected x_train with shape (3, batch_size, time_steps, feature_dim)')
    nb_train_sample = 0
    for c in range(0, len(x_train)):
        x_train[c] = pad_sequences(x_train[c], maxlen=max_len, dtype=np.float32, padding='pre', truncating='pre')
        print('train shape: ', (len(x_train[c]), x_train[c][0].shape))
        nb_train_sample = x_train[c].shape[0]

    for c in range(0, len(x_test)):
        x_test[c] = pad_sequences(x_test[c], maxlen=max_len, dtype=np.float32, padding='pre', truncating='pre')
        print('test shape:  ', x_test[c].shape)

    y_train = np_utils.to_categorical(y_train, class_num)
    y_test = np_utils.to_categorical(y_test, class_num)

    print('Train...')
    save_dir = 'snapshot/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    snapshot_file = save_dir + '/weights_' + save_postfix + '_testid_{}'.format(test_id) + '.hdf5'

    def scheduler(epoch):
        lr = np.float32(K.get_value(model.optimizer.lr))
        print('learning rate: {}'.format(lr))
        if epoch == 60 or epoch == 80 or epoch == 90:
            print('epoch: {}, learning rate divided by 10'.format(epoch))
            lr = np.float32(K.get_value(model.optimizer.lr) / 10.0)
            K.set_value(model.optimizer.lr, lr)
        return lr

    changelr = LearningRateScheduler(scheduler)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch,
              validation_data=(x_test, y_test), verbose=1,
              callbacks=[changelr])
    model.save_weights(snapshot_file)
    # restore the final model
    model.load_weights(snapshot_file)
    # test
    print('Test...')
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:    ', score)
    print('Test accuracy: ', acc)
    # save predictions
    pred_result = model.predict(x_test, batch_size=batch_size, verbose=1)
    save_dir = 'result/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sio.savemat(save_dir + '/pred_results_'+save_postfix+'_testid_{}'.format(test_id)+'.mat', {'pred_result': pred_result})
    print(save_dir + '/pred_results_'+save_postfix+'_testid_{}'.format(test_id)+'.mat saved!')
    with open(save_dir+'/result.txt', 'a') as f:
        f.write('{}\t{}\n'.format(test_id, acc))
   
    return model


def test_model(model, x_test, y_test, class_num, batch_size, save_postfix, MAX_LEN, test_id):
    for c in range(0, len(x_test)):
        x_test[c] = pad_sequences(x_test[c], maxlen=MAX_LEN, dtype=np.float32, padding='pre', truncating='pre')
        print('test shape:  ', x_test[c].shape)    # x_test = pad_sequences(x_test, maxlen=1MAX_LEN)
    y_test = np_utils.to_categorical(y_test, class_num)
    # load the snapshot model
    save_dir = 'snapshot/'+save_postfix
    snapshot_file = save_dir + '/weights_' + save_postfix + '_testid_{}'.format(test_id) + '.hdf5'
    # save_dir = 'snapshot/'
    # snapshot_file = save_dir + 'weights_' + save_postfix + '.hdf5'
    print('loading {}'.format(snapshot_file))
    model.load_weights(snapshot_file)
    # test
    print('Test...')
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:    ', score)
    print('Test accuracy: ', acc)
    # save predictions
    pred_result = model.predict(x_test, batch_size=batch_size, verbose=1)
    save_dir = 'result/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sio.savemat(save_dir + '/pred_results_'+save_postfix+'_testid_{}'.format(test_id)+'.mat', {'pred_result': pred_result})
    print(save_dir + '/pred_results_'+save_postfix+'_testid_{}'.format(test_id)+'.mat saved!')
    with open(save_dir+'/result.txt', 'a') as f:
        f.write('{}\t{}\n'.format(test_id, acc))

def visualize_model(model, save_dir):
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file=save_dir + '/model.png')


if __name__ == '__main__':
    from keras.optimizers import RMSprop, Adam
    from keras.layers.recurrent import LSTM
    #get_model(66, [(LSTM, 100, 0.5)] * 3, [(256, 0.5)] * 2, 28, True,
    #        RMSprop(lr=0.001))
    rnn_cells = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    rnn_cells2 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    rnn_cells3 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    fc_dims = [(512, 0.5), (256, 0.5)]
    model = get_model_three_branches_new(100, rnn_cells, 30, rnn_cells2, 66, rnn_cells3, fc_dims, 14,
            False, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    visualize_model(model, '.')
