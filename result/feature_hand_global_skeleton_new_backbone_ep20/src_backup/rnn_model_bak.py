from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense, Lambda
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
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
from gesture_dataset import GestureDataGenerator

# MAX_LEN = 64


def get_model(input_dim, rnn_cells, fc_dims, output_dim, is_bid, optimizer):
    model = Sequential()
    # RNN layers
    for idx, rnn_cell in enumerate(rnn_cells):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True)  
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim))
        model.add(layer)
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(Dropout(0.5))

    # FC layers
    for fc_dim, dropout_rate in fc_dims:
        model.add(TimeDistributed(Dense(fc_dim, activation='relu')))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Lambda(function=lambda x: K.mean(x, axis=1), 
                     output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def get_model_two_branches(input_dim1, rnn_cells1, input_dim2, rnn_cells2, fc_dims, output_dim, is_bid, optimizer):
    model = Sequential()
    upper_branch = Sequential()
    lower_branch = Sequential()
    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim1)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        upper_branch.add(layer)
        if dropout_rate > 0:
            upper_branch.add(Dropout(dropout_rate))
    upper_branch.add(TimeDistributed(Dense(256, activation='relu')))
    upper_branch.add(Dropout(0.5))

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim2)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        lower_branch.add(layer)
        if dropout_rate > 0:
            lower_branch.add(Dropout(dropout_rate))
    lower_branch.add(TimeDistributed(Dense(256, activation='relu')))
    lower_branch.add(Dropout(0.5))

    # Merge
    merged = Merge([upper_branch, lower_branch], mode='concat')
    model.add(merged)

    # FC layers
    for fc_dim, dropout_rate in fc_dims:
        model.add(TimeDistributed(Dense(fc_dim, activation='relu')))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Lambda(function=lambda x: K.mean(x, axis=1),
                     output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def get_model_three_branches(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
    model = Sequential()
    upper_branch = Sequential()
    lower_branch = Sequential()
    middle_branch = Sequential()
    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim1)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        upper_branch.add(layer)
        if dropout_rate > 0:
            upper_branch.add(Dropout(dropout_rate))
    upper_branch.add(TimeDistributed(Dense(256, activation='relu')))
    upper_branch.add(Dropout(0.5))

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim2)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        lower_branch.add(layer)
        if dropout_rate > 0:
            lower_branch.add(Dropout(dropout_rate))
    lower_branch.add(TimeDistributed(Dense(256, activation='relu')))
    lower_branch.add(Dropout(0.5))

    # Middle RNN layers
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim3)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim3))
        middle_branch.add(layer)
        if dropout_rate > 0:
            middle_branch.add(Dropout(dropout_rate))
    middle_branch.add(TimeDistributed(Dense(256, activation='relu')))
    middle_branch.add(Dropout(0.5))

    # Merge
    merged = Merge([upper_branch, lower_branch, middle_branch], mode='concat')
    model.add(merged)

    # FC layers
    for fc_dim, dropout_rate in fc_dims:
        model.add(TimeDistributed(Dense(fc_dim, activation='relu')))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Lambda(function=lambda x: K.mean(x, axis=1),
                     output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def get_model_three_branches_stack_lstm(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
    model = Sequential()
    upper_branch = Sequential()
    lower_branch = Sequential()
    middle_branch = Sequential()
    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim1), kernel_initializer='he_normal', recurrent_dropout=0.25)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        upper_branch.add(layer)
        # upper_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            upper_branch.add(Dropout(dropout_rate))

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim2), kernel_initializer='he_normal', recurrent_dropout=0.25)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        lower_branch.add(layer)
        # lower_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            lower_branch.add(Dropout(dropout_rate))
        lower_branch.add(BatchNormalization())

    # Middle RNN layers
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim3), kernel_initializer='he_normal', recurrent_dropout=0.25)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim3))
        middle_branch.add(layer)
        # middle_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            middle_branch.add(Dropout(dropout_rate))
        middle_branch.add(BatchNormalization())

    # Merge
    merged = Concatenate()([upper_branch, lower_branch, middle_branch])
    model.add(merged)

    layer = LSTM(1024, kernel_initializer='he_normal', recurrent_dropout=0.5)
    model.add(layer)
    model.add(BatchNormalization())

    # FC layers
    # for fc_dim, dropout_rate in fc_dims:
    for idx, fc_cell in enumerate(fc_dims):
        fc_dim, dropout_rate = fc_cell
        # model.add(Dense(fc_dim, activation='relu'))
        model.add(Dense(fc_dim))#, init='normal'))
        if idx < len(fc_dims)-1:
            #model.add(BatchNormalization())
            model.add(Activation('relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim))#, init='normal'))
    # model.add(Lambda(function=lambda x: K.mean(x, axis=1),
    #                 output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def get_model_three_branches_new(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
    model = Sequential()
    upper_branch = Sequential()
    lower_branch = Sequential()
    middle_branch = Sequential()
    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx == len(rnn_cells1)-1:
            layer = cell_type(cell_num, kernel_initializer='he_normal', recurrent_dropout=0.25)
        elif idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim1, kernel_initializer='he_normal', recurrent_dropout=0.25)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        upper_branch.add(layer)
        # upper_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            upper_branch.add(Dropout(dropout_rate))
    # upper_branch.add(Dense(512, activation='relu'))
    # upper_branch.add(Dense(512, init='normal'))
    # upper_branch.add(BatchNormalization())
    # upper_branch.add(Activation('relu'))
    # upper_branch.add(Dropout(0.5))

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx == len(rnn_cells2)-1:
            layer = cell_type(cell_num, kernel_initializer='he_normal', recurrent_dropout=0.25)
        elif idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim2, kernel_initializer='he_normal', recurrent_dropout=0.25)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        lower_branch.add(layer)
        # lower_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            lower_branch.add(Dropout(dropout_rate))
        lower_branch.add(BatchNormalization())
    # lower_branch.add(Dense(512, activation='relu'))
    # lower_branch.add(Dense(512, init='normal'))
    # lower_branch.add(BatchNormalization())
    # lower_branch.add(Activation('relu'))
    # lower_branch.add(Dropout(0.5))

    # Middle RNN layers
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx == len(rnn_cells3)-1:
            layer = cell_type(cell_num, kernel_initializer='he_normal', recurrent_dropout=0.25)
        elif idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_dim=input_dim3, kernel_initializer='he_normal', recurrent_dropout=0.25)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim3))
        middle_branch.add(layer)
        # middle_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            middle_branch.add(Dropout(dropout_rate))
        middle_branch.add(BatchNormalization())
    # middle_branch.add(Dense(512, activation='relu'))
    # middle_branch.add(Dense(512, init='normal'))
    # middle_branch.add(BatchNormalization())
    # middle_branch.add(Activation('relu'))
    # middle_branch.add(Dropout(0.5))

    # Merge
    merged = Merge([upper_branch, lower_branch, middle_branch], mode='concat')
    model.add(merged)

    # FC layers
    # for fc_dim, dropout_rate in fc_dims:
    for idx, fc_cell in enumerate(fc_dims):
        fc_dim, dropout_rate = fc_cell
        model.add(Dense(fc_dim, activation='relu'))
        # model.add(Dense(fc_dim))#, init='normal'))
        # if idx < len(fc_dims)-1:
        #     #model.add(BatchNormalization())
        #     model.add(Activation('relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim))#, init='normal'))
    # model.add(Lambda(function=lambda x: K.mean(x, axis=1),
    #                 output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def _pad_sequences(seqs, max_len = -1):
    """
    currently deprecated. use keras.preprocessing.sequence.pad_sequences instead
    :param seqs: 
    :param max_len: 
    :return: 
    """
    if max_len < 0:
        lengths = [s.shape[0] for s in seqs]
        max_len = np.max(lengths)
    seq_arr = np.zeros([len(seqs), max_len, seqs[0].shape[1]], dtype=np.float32)

    for idx, s in enumerate(seqs):
        seq_arr[idx, :s.shape[0], :] = s

    return seq_arr


def train_model(model, x_train, y_train, x_test, y_test, class_num, batch_size,
                num_epoch, save_postfix, max_len, save_best_only_, test_id, data_generator=None, is_data_augmented=False):
    if is_data_augmented and len(x_train) != 1:
        raise ValueError('Expected x_train with shape (batch_size, time_steps, feature_dim)')
    if not is_data_augmented and len(x_train) != 3:
        raise ValueError('Expected x_train with shape (3, batch_size, time_steps, feature_dim)')
    nb_train_sample = 0
    if not is_data_augmented:
        for c in range(0, len(x_train)):
            x_train[c] = pad_sequences(x_train[c], maxlen=max_len, dtype=np.float32, padding='pre', truncating='pre')
            print('train shape: ', (len(x_train[c]), x_train[c][0].shape))
            nb_train_sample = x_train[c].shape[0]
    else:
        for c in range(0, len(x_train)):
            print('train shape: ', x_train[c].shape)
            nb_train_sample = x_train[c].shape[0]
    for c in range(0, len(x_test)):
        x_test[c] = pad_sequences(x_test[c], maxlen=max_len, dtype=np.float32, padding='pre', truncating='pre')
        print('test shape:  ', x_test[c].shape)

    # x_train = _PadSequences(x_train)
    # x_test = _PadSequences(x_test, max_len=x_train.shape[1])
    #print x_train.shape

    y_train = np_utils.to_categorical(y_train, class_num)
    y_test = np_utils.to_categorical(y_test, class_num)

    print('Train...')
    save_dir = 'snapshot/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    snapshot_file = save_dir + '/weights_' + save_postfix + '_testid_{}'.format(test_id) + '.hdf5'
    checkpointer = ModelCheckpoint(
            filepath=snapshot_file, monitor='val_acc',
            verbose=1, save_best_only=save_best_only_)

    def scheduler(epoch):
        lr = np.float32(K.get_value(model.optimizer.lr))
        print('learning rate: {}'.format(lr))
        if epoch > 0 and epoch % 50 == 0:
            print('epoch: {}, learning rate divided by 10'.format(epoch))
            lr = np.float32(K.get_value(model.optimizer.lr) / 10.0)
            K.set_value(model.optimizer.lr, lr)
        #print('shape:   ', lr.shape)
        #print('dtype:   ', lr.dtype)
        return lr

    changelr = LearningRateScheduler(scheduler)

    if is_data_augmented:
        model.fit_generator(data_generator.flow(x_train[0], y_train, batch_size=batch_size, max_len=max_len),
                            samples_per_epoch=nb_train_sample, nb_epoch=num_epoch,
                            validation_data=(x_test, y_test), verbose=1,
                            callbacks=[checkpointer, changelr],
                            max_q_size=6, nb_worker=6, pickle_safe=True)
    else:
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
                  validation_data=(x_test, y_test), verbose=1,
                  callbacks=[checkpointer, changelr])

    # restore the best model
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

    return model


def train_model_online_feature(model, x_train, y_train, x_test, y_test, class_num, batch_size,
                               num_epoch, save_postfix, MAX_LEN, save_best_only_, test_id, data_generator=None):
    for c in range(0, len(x_train)):
        # x_train[c] = pad_sequences(x_train[c], maxlen=MAX_LEN, dtype=np.float32, padding='pre', truncating='pre')
        print('train shape: ', len(x_train[c]))
        nb_train_sample = len(x_train[c])
    for c in range(0, len(x_test)):
        x_test[c] = pad_sequences(x_test[c], maxlen=MAX_LEN, dtype=np.float32, padding='pre', truncating='pre')
        print('test shape:  ', x_test[c].shape)

    # x_train = _PadSequences(x_train)
    # x_test = _PadSequences(x_test, max_len=x_train.shape[1])
    #print x_train.shape

    y_train = np_utils.to_categorical(y_train, class_num)
    y_test = np_utils.to_categorical(y_test, class_num)

    print('Train...')
    save_dir = 'snapshot/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    snapshot_file = save_dir + '/weights_' + save_postfix + '_testid_{}'.format(test_id) + '.hdf5'
    checkpointer = ModelCheckpoint(
            filepath=snapshot_file, monitor='val_acc',
            verbose=1, save_best_only=save_best_only_)

    def scheduler(epoch):
        lr = np.float32(K.get_value(model.optimizer.lr))
        print('learning rate: {}'.format(lr))
        if epoch > 0 and epoch % 50 == 0:
            print('epoch: {}, learning rate divided by 10'.format(epoch))
            lr = np.float32(K.get_value(model.optimizer.lr) / 10.0)
            K.set_value(model.optimizer.lr, lr)
        #print('shape:   ', lr.shape)
        #print('dtype:   ', lr.dtype)
        return lr

    changelr = LearningRateScheduler(scheduler)

    # model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
    #         validation_data=(x_test, y_test), verbose=1,
    #         #callbacks=[checkpointer])
    #         callbacks=[checkpointer, changelr])
    model.fit_generator(data_generator.flow(x_train[0], y_train, batch_size=batch_size, max_len = MAX_LEN),
                        samples_per_epoch=nb_train_sample, nb_epoch=num_epoch,
                        validation_data=(x_test, y_test), verbose=1,
                        callbacks=[checkpointer, changelr],
                        max_q_size=6, nb_worker=6, pickle_safe=True)

    # restore the best model
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

    return model


def test_model(model, x_test, y_test, class_num, batch_size, save_postfix, MAX_LEN, test_id):
    for c in range(0, len(x_test)):
        x_test[c] = pad_sequences(x_test[c], maxlen=MAX_LEN, dtype=np.float32, padding='post', truncating='pre')
        print('test shape:  ', x_test[c].shape)    # x_test = pad_sequences(x_test, maxlen=1MAX_LEN)
    y_test = np_utils.to_categorical(y_test, class_num)
    # load the snapshot model
    save_dir = 'snapshot/'+save_postfix
    snapshot_file = save_dir + '/weights_' + save_postfix + '_testid_{}'.format(test_id) + '.hdf5'
    # save_dir = 'snapshot/'
    # snapshot_file = save_dir + 'weights_' + save_postfix + '.hdf5'
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


def visualize_model(model, save_dir):
    from keras.utils.visualize_util import plot
    plot(model, to_file=save_dir + '/model.png')


if __name__ == '__main__':
    from keras.optimizers import RMSprop, Adam
    from keras.layers.recurrent import LSTM
    #get_model(66, [(LSTM, 100, 0.5)] * 3, [(256, 0.5)] * 2, 28, True,
    #        RMSprop(lr=0.001))
    rnn_cells = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    rnn_cells2 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    rnn_cells3 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    fc_dims = [(512, 0.5), (256, 0.5)]
    model = get_model_three_branches(100, rnn_cells, 30, rnn_cells2, 66, rnn_cells3, fc_dims, 14,
            1, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    visualize_model(model, '.')
