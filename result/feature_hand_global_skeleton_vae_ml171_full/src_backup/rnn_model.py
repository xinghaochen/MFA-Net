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
from gesture_dataset import GestureDataGenerator
from keras.utils import multi_gpu_model

import tensorflow as tf


# MAX_LEN = 64

np.random.seed(10000)

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
    upper_input = Input(shape=(None, input_dim1), dtype='float32', name='upper_input')
    lower_input = Input(shape=(None, input_dim2), dtype='float32', name='lower_input')
    middle_input = Input(shape=(None, input_dim3), dtype='float32', name='middle_input')

    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim1), kernel_initializer='he_normal', recurrent_dropout=0.25)(upper_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        # upper_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            upper_branch = Dropout(dropout_rate)(layer)
        else:
            upper_branch = layer            
    upper_branch = TimeDistributed(Dense(256, activation='relu'))(upper_branch)
    upper_branch = Dropout(0.5)(upper_branch)

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim2), kernel_initializer='he_normal', recurrent_dropout=0.25)(lower_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        # lower_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            lower_branch = Dropout(dropout_rate)(layer)
        else:
            lower_branch = layer
    lower_branch = TimeDistributed(Dense(256, activation='relu'))(lower_branch)
    lower_branch = Dropout(0.5)(lower_branch)
    
    # Middle RNN layers
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim3), kernel_initializer='he_normal', recurrent_dropout=0.25)(middle_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim3))
        # middle_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            middle_branch = Dropout(dropout_rate)(layer)
        else:
            middle_branch = layer
    middle_branch = TimeDistributed(Dense(256, activation='relu'))(middle_branch)
    middle_branch = Dropout(0.5)(middle_branch)
    
    # Merge
    merged = concatenate([upper_branch, lower_branch, middle_branch])
    layer = merged
    
    # FC layers
    for idx, fc_cell in enumerate(fc_dims):
        fc_dim, dropout_rate = fc_cell
        layer = Dense(fc_dim)(layer)#, init='normal'))
        if idx < len(fc_dims)-1:
            layer = Activation('relu')(layer)
        if dropout_rate > 0:
            layer  =Dropout(dropout_rate)(layer)
    layer = Dense(output_dim)(layer)#, init='normal'))
    layer = Lambda(function=lambda x: K.mean(x, axis=1),
                     output_shape=lambda shape: (shape[0],) + shape[2:])(layer)
    main_output = Activation('softmax')(layer)
    model = Model(inputs=[upper_input, lower_input, middle_input], outputs=[main_output])


    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def get_model_three_branches_stack_lstm(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
    upper_input = Input(shape=(None, input_dim1), dtype='float32', name='upper_input')
    lower_input = Input(shape=(None, input_dim2), dtype='float32', name='lower_input')
    middle_input = Input(shape=(None, input_dim3), dtype='float32', name='middle_input')

    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim1), kernel_initializer='he_normal', recurrent_dropout=0.25)(upper_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        # upper_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            upper_branch = Dropout(dropout_rate)(layer)
        else:
            upper_branch = layer

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim2), kernel_initializer='he_normal', recurrent_dropout=0.25)(lower_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        # lower_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            lower_branch = Dropout(dropout_rate)(layer)
        else:
            lower_branch = layer

    # Middle RNN layers
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim3), kernel_initializer='he_normal', recurrent_dropout=0.25)(middle_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim3))
        # middle_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            middle_branch = Dropout(dropout_rate)(layer)
        else:
            middle_branch = layer

    # Merge
    merged = concatenate([upper_branch, lower_branch, middle_branch])

    layer = LSTM(1024, kernel_initializer='he_normal', recurrent_dropout=0.5)(merged)

    # FC layers
    # for fc_dim, dropout_rate in fc_dims:
    for idx, fc_cell in enumerate(fc_dims):
        fc_dim, dropout_rate = fc_cell
        # model.add(Dense(fc_dim, activation='relu'))
        layer = Dense(fc_dim)(layer)#, init='normal'))
        if idx < len(fc_dims)-1:
            #model.add(BatchNormalization())
            layer = Activation('relu')(layer)
        if dropout_rate > 0:
            layer  =Dropout(dropout_rate)(layer)
    layer = Dense(output_dim)(layer)#, init='normal'))
    # model.add(Lambda(function=lambda x: K.mean(x, axis=1),
    #                 output_shape=lambda shape: (shape[0],) + shape[2:]))
    main_output = Activation('softmax')(layer)
    with tf.device('/cpu:0'):
        model = Model(inputs=[upper_input, lower_input, middle_input], outputs=[main_output])
    model._make_predict_function()

    # Replicates the model on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    #parallel_model = multi_gpu_model(model, gpus=[0,1,2,3])
    parallel_model = model

    parallel_model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    graph = None#K.get_session().graph
    return parallel_model, graph

def get_model_three_branches_stack_lstm_new(input_dim1, input_dim2, input_dim3, fc_dims, output_dim, is_bid, optimizer):
    upper_input = Input(shape=(None, input_dim1), dtype='float32', name='upper_input')
    lower_input = Input(shape=(None, input_dim2), dtype='float32', name='lower_input')
    middle_input = Input(shape=(None, input_dim3), dtype='float32', name='middle_input')

    # Upper RNN layers
    for idx, rnn_cell in enumerate(rnn_cells1):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim1), kernel_initializer='he_normal', recurrent_dropout=0.25)(upper_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim1))
        upper_branch.add(layer)
        # upper_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            upper_branch = Dropout(dropout_rate)(layer)
        else:
            upper_branch = layer

    # Lower RNN layers
    for idx, rnn_cell in enumerate(rnn_cells2):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim2), kernel_initializer='he_normal', recurrent_dropout=0.25)(lower_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim2))
        # lower_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            lower_branch = Dropout(dropout_rate)(layer)
        else:
            lower_branch = layer

    # Middle RNN layers
    for idx, rnn_cell in enumerate(rnn_cells3):
        cell_type, cell_num, dropout_rate = rnn_cell
        if idx:
            layer = cell_type(cell_num, return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.25)(layer)
        else:
            layer = cell_type(cell_num, return_sequences=True, input_shape=(None, input_dim3), kernel_initializer='he_normal', recurrent_dropout=0.25)(middle_input)
        if is_bid:
            layer = Bidirectional(layer) if idx else Bidirectional(layer, input_shape=(None, input_dim3))
        middle_branch.add(layer)
        # middle_branch.add(GaussianNoise(0.25))
        if dropout_rate > 0:
            middle_branch = Dropout(dropout_rate)(layer)
        else:
            middle_branch = layer

    # Merge
    merged = concatenate([upper_branch, lower_branch, middle_branch])
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


def get_model_three_branches_new_dropout(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
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
        if idx < len(rnn_cells1) - 1:
            upper_branch = SpatialDropout1D(0.25)(upper_branch)

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
        if idx < len(rnn_cells2) - 1:
            lower_branch = SpatialDropout1D(0.25)(lower_branch)
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
        if idx < len(rnn_cells3) - 1:
            middle_branch = SpatialDropout1D(0.25)(middle_branch)
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


# https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
def attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR = False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    time_steps = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def get_model_three_branches_new_attention(input_dim1, rnn_cells1, input_dim2, rnn_cells2, input_dim3, rnn_cells3, fc_dims, output_dim, is_bid, optimizer):
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
    upper_branch = attention_3d_block(upper_branch)
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
    lower_branch = attention_3d_block(lower_branch)
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
    middle_branch = attention_3d_block(middle_branch)
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
            verbose=1, save_best_only=save_best_only_)#, period = 10)

    def scheduler(epoch):
        lr = np.float32(K.get_value(model.optimizer.lr))
        print('learning rate: {}'.format(lr))
        #print('epoch: {}, learning rate divided by 10'.format(epoch))
#        if epoch > 0 and epoch % 5 == 0:
#            lr = np.float32(K.get_value(model.optimizer.lr) *0.7)
#            K.set_value(model.optimizer.lr, lr)
#         if epoch > 0 and epoch % 50 == 0:
        if epoch == 60 or epoch == 80 or epoch == 90:
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
        model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch,
                  validation_data=(x_test, y_test), verbose=1,
                  #callbacks=[checkpointer])
                  callbacks=[changelr])
    model.save_weights(snapshot_file)
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
    with open(save_dir+'/result.txt', 'a') as f:
        f.write('{}\t{}\n'.format(test_id, acc))
   
    return model


def train_model_online_feature(model, graph, x_train, y_train, x_test, y_test, class_num, batch_size,
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
    #with graph.as_default():
    model.fit_generator(data_generator.generate_with_vae_feature(x_train[0], y_train, batch_size=batch_size, max_len = MAX_LEN),
                        steps_per_epoch=nb_train_sample/batch_size, epochs=num_epoch,
                        validation_data=(x_test, y_test),
                        verbose=1, callbacks=[changelr],
                        max_queue_size=20, workers=50, use_multiprocessing=False)

    model.save_weights(snapshot_file)
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
