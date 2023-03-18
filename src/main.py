import argparse
import gesture_dataset_DHG2016
import gesture_dataset_shrec17_aug as gesture_dataset_shrec17
import rnn_model
import time
import os

# set gpu using tf
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.recurrent import LSTM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', help='root directory for dataset')
    parser.add_argument('-dataset', '--dataset', help='the name of dataset', default='DHG2016')
    parser.add_argument('-i', '--id', help='subject id for test',
                        type=int)
    parser.add_argument('-f', '--full', help='True to test 28 classes',
                        type=int, default=0)
    parser.add_argument('-l', '--lr', help='learning rate', type=float)
    parser.add_argument('-b', '--batch_size', help='batch_size', type=int)
    parser.add_argument('-e', '--num_epoch', help='number of epoch to train',
                        type=int)
    parser.add_argument('-s', '--snapshot_postfix', help='snapshot postfix')
    parser.add_argument('-m', '--mode', help='training mode or testing mode',
                        type=int, default=0)
    parser.add_argument('-mamp', '--M_amp', help='M_amp',
                        type=int, default=5)
    parser.add_argument('-mdf', '--max_dist_factor', help='max_dist_factor',
                        type=float, default=1.5)
    parser.add_argument('-of1', '--offset1', help='offset1',
                        type=int, default=5)
    parser.add_argument('-of2', '--offset2', help='offset2',
                        type=int, default=10)
    parser.add_argument('-ml', '--MAX_LEN', help='MAX_LEN',
                        type=int, default=64)
    parser.add_argument('-bi', '--is_bid', dest='is_bid', action='store_true')
    parser.set_defaults(is_bid=False)

    args = parser.parse_args()
    print args
    return (args.root_dir, args.id, args.full, args.lr, args.batch_size,
            args.num_epoch, args.snapshot_postfix, args.mode,
            args.M_amp, args.max_dist_factor, args.offset1, args.offset2,
            args.MAX_LEN, args.is_bid, args.dataset)


def get_three_branches_model(input_dim, class_num, lr, is_bid):
    rnn_cells = [(LSTM, 100, 0.5), (LSTM, 100, 0.5)]
    rnn_cells2 = [(LSTM, 100, 0.3), (LSTM, 100, 0.3)]
    rnn_cells3 = [(LSTM, 100, 0.3), (LSTM, 100, 0.3)]
    fc_dims = [(128, 0.5)]
    return rnn_model.get_model_three_branches_new(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, input_dim[2], rnn_cells3, fc_dims, class_num,
             is_bid=False, optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))


def main(root_dir, test_id, is_full, lr, batch_size, num_epoch, save_postfix, mode,
         M, max_dist_factor, offset1, offset2, MAX_LEN, is_bid, dataset):
    # mkdir
    # save_dir = 'result/'+save_postfix
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # save_dir = 'snapshot/'+save_postfix
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # save_dir = 'result/'+save_postfix+'/src_backup'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    start_time = time.time()
    print 'is_bid: {}'.format(is_bid)

    if dataset == 'SHREC17':
        data = gesture_dataset_shrec17.Dataset(root_dir, is_full, is_aug=(mode==0))
        (x_train_all, y_train), (x_test_all, y_test) = \
            data.load_data_with_vae_feature(M, max_dist_factor, offset1, offset2, is_preprocess=True, load_test_only=(mode==1))
    elif dataset == 'DHG2016':
        data = gesture_dataset_DHG2016.Dataset(root_dir, is_full)
        (x_train_all, y_train), (x_test_all, y_test) = \
            data.load_data_with_vae_feature(test_id, M, max_dist_factor, offset1, offset2, is_preprocess=True, load_test_only=(mode == 1))

    print 'elapsed time: {} s.'.format(time.time() - start_time)
    start_time = time.time()

    model = get_three_branches_model(data.input_dim, data.class_num, lr, is_bid)

    print 'elapsed time: {} s.'.format(time.time() - start_time)
    print 'Finish loading data!'
    print 'test_id: {}'.format(test_id)

    if mode == 0:
        rnn_model.train_model(model, x_train_all, y_train, x_test_all, y_test,
                              data.class_num, batch_size, num_epoch, save_postfix, MAX_LEN, test_id)
    elif mode == 1:
        rnn_model.test_model(model, x_test_all, y_test,
                            data.class_num, batch_size, save_postfix, MAX_LEN, test_id)


if __name__ == '__main__':
    main(*parse_arguments())
