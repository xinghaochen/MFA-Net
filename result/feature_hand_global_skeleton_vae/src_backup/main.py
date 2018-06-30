import argparse
import gesture_dataset
import gesture_dataset_shrec17
from gesture_dataset import GestureDataGenerator
import rnn_model
import feature_extractor
import time
import os

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
    parser.add_argument('-data', '--data', help='data using for the network',
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
    # parser.add_argument('-bi', '--is_bid', help='is bidirectional',
    #                     type=bool, default=False)
    # parser.add_argument('-sbo', '--save_best_only', help='save_best_only',
    #                     type=bool, default=True)
    parser.add_argument('-bi', '--is_bid', dest='is_bid', action='store_true')
    parser.set_defaults(is_bid=False)
    parser.add_argument('-nsbo', '--no-save_best_only', dest='save_best_only', action='store_false')
    parser.set_defaults(save_best_only=True)

    args = parser.parse_args()
    return (args.root_dir, args.id, args.full, args.lr, args.batch_size,
            args.num_epoch, args.snapshot_postfix, args.mode, args.data,
            args.M_amp, args.max_dist_factor, args.offset1, args.offset2,
            args.MAX_LEN, args.is_bid, args.save_best_only, args.dataset)


def get_baseline_model(input_dim, class_num, lr, is_bid):
    rnn_cells = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    fc_dims = [(512, 0.5), (256, 0.5)]
    # rnn_cells = [(LSTM, 64, 0.3), (LSTM, 128, 0.3), (LSTM, 128, 0.3)]
    # fc_dims = [(128, 0.5), (128, 0.5)]
    # return rnn_model.get_model(input_dim, rnn_cells, fc_dims, class_num,
    #                is_bid=False, optimizer=RMSprop(lr=lr))
    return rnn_model.get_model(input_dim, rnn_cells, fc_dims, class_num,
             is_bid, optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))


def get_two_branches_model(input_dim, class_num, lr, is_bid):
    rnn_cells = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    rnn_cells2 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    fc_dims = [(512, 0.5), (256, 0.5)]
    # rnn_cells = [(LSTM, 64, 0.3), (LSTM, 128, 0.3), (LSTM, 128, 0.3)]
    # fc_dims = [(128, 0.5), (128, 0.5)]
    # return rnn_model.get_model(input_dim, rnn_cells, fc_dims, class_num,
    #                is_bid=False, optimizer=RMSprop(lr=lr))
    return rnn_model.get_model_two_branches(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, fc_dims, class_num,
             is_bid, optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    # return rnn_model.get_model_two_branches(100, rnn_cells, 25, rnn_cells2, fc_dims, class_num,
    #             is_bid=False, optimizer=RMSprop(lr=lr))


def get_three_branches_model(input_dim, class_num, lr, is_bid):
    rnn_cells = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    rnn_cells2 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    rnn_cells3 = [(LSTM, 128, 0.3), (LSTM, 256, 0.3)]
    fc_dims = [(512, 0.5), (256, 0.5)]
    # rnn_cells = [(LSTM, 64, 0.3), (LSTM, 128, 0.3), (LSTM, 128, 0.3)]
    # fc_dims = [(128, 0.5), (128, 0.5)]
    # return rnn_model.get_model(input_dim, rnn_cells, fc_dims, class_num,
    #                is_bid=False, optimizer=RMSprop(lr=lr))
    return rnn_model.get_model_three_branches(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, 66, rnn_cells3, fc_dims, class_num,
             is_bid=False, optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
#    return rnn_model.get_model_three_branches(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, 66, rnn_cells3, fc_dims, class_num,
#             is_bid=False, optimizer=RMSprop(lr=lr))
    # return rnn_model.get_model_two_branches(100, rnn_cells, 25, rnn_cells2, fc_dims, class_num,
    #             is_bid=False, optimizer=RMSprop(lr=lr))


def get_three_branches_model_new(input_dim, class_num, lr, is_bid):
    rnn_cells = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    rnn_cells2 = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    rnn_cells3 = [(LSTM, 256, 0.5), (LSTM, 512, 0.5)]
    #fc_dims = [(128, 0.5)]
    # rnn_cells = [(LSTM, 64, 0.3), (LSTM, 128, 0.3), (LSTM, 128, 0.3)]
    fc_dims = [(1024, 0.5)]
    # return rnn_model.get_model(input_dim, rnn_cells, fc_dims, class_num,
    #                is_bid=False, optimizer=RMSprop(lr=lr))
    print input_dim
    return rnn_model.get_model_three_branches_stack_lstm(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, input_dim[2], rnn_cells3, fc_dims, class_num,
                                           is_bid, optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    #return rnn_model.get_model_three_branches_new(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, 66, rnn_cells3, fc_dims, class_num,
    #            is_bid, optimizer=RMSprop(lr=lr, decay=0.1))
    #return rnn_model.get_model_three_branches(input_dim[0], rnn_cells, input_dim[1], rnn_cells2, 66, rnn_cells3, fc_dims, class_num,
    #             is_bid, optimizer=SGD(lr=lr, momentum=0.9, decay=0.0))


def main(root_dir, test_id, is_full, lr, batch_size, num_epoch, save_postfix, mode, data,
         M, max_dist_factor, offset1, offset2, MAX_LEN, is_bid, save_best_only, dataset):
    # mkdir
    save_dir = 'result/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = 'snapshot/'+save_postfix
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = 'result/'+save_postfix+'/src_backup'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # backup all files
    cmd = 'find . -name \'*.py\' -exec cp {} result/'+ save_postfix +'/src_backup/ \\;'
    print cmd
    os.system(cmd)
    
    ## 
    do_pca = 0
    pca_dim = 24
    is_normalize = 0
    start_time = time.time()
    print 'is_bid: {}'.format(is_bid)
    if data == 0:
        data = gesture_dataset.Dataset(root_dir, is_full)
        (x_train, y_train), (x_test, y_test) = data.load_data(test_id, is_preprocess=False)
        x_train_all = [x_train]
        x_test_all = [x_test]
        model = get_baseline_model(data.input_dim, data.class_num, lr, is_bid)
    elif data == 1:
        data = gesture_dataset.Dataset(root_dir, is_full)
        (x_train, y_train), (x_test, y_test) = data.load_data(test_id, is_preprocess=True)
        x_train_all = [x_train]
        x_test_all = [x_test]
        model = get_baseline_model(data.input_dim, data.class_num, lr, is_bid)
    elif data == 2:
        data = feature_extractor.FeatureExtractor("../data/", is_full)
        (x_train, y_train), (x_test, y_test) = data.load_data()
        x_train_all = [x_train]
        x_test_all = [x_test]
        model = get_baseline_model(data.input_dim, data.class_num, lr, is_bid)
    elif data == 3: # two branch data
        data = feature_extractor.FeatureExtractor("../data/", is_full)
        x_train, x_train2, y_train = data.extract_feature(test_id, 0, M, max_dist_factor, offset1, offset2)
        x_test, x_test2, y_test = data.extract_feature(test_id, 1, M, max_dist_factor, offset1, offset2)
        # (x_train, x_train2, y_train), (x_test, x_test2, y_test) = data.load_data2()
        x_train_all = [x_train, x_train2]
        x_test_all = [x_test, x_test2]
        model = get_two_branches_model(data.input_dim, data.class_num, lr, is_bid)
    elif data == 4: # three branch data
        data = gesture_dataset.Dataset(root_dir, is_full)
        (x_train3, y_train), (x_test3, y_test) = data.load_data(test_id, is_preprocess=True)
        data = feature_extractor.FeatureExtractor("../data/", is_full)
        x_train, x_train2, y_train = data.extract_feature(test_id, 0, M, max_dist_factor, offset1, offset2)
        x_test, x_test2, y_test = data.extract_feature(test_id, 1, M, max_dist_factor, offset1, offset2)
        # (x_train, x_train2, y_train), (x_test, x_test2, y_test) = data.load_data2()
        x_train_all = [x_train, x_train2, x_train3]
        x_test_all = [x_test, x_test2, x_test3]
        model = get_three_branches_model(data.input_dim, data.class_num, lr, is_bid)
    elif data == 5: # preload data and feature, batch generator
#        data = gesture_dataset.Dataset(root_dir, is_full)
#        (x_train_all, y_train), (x_test_all, y_test) = data.load_data_with_feature(test_id, do_pca, pca_dim, M, max_dist_factor,
#                                                                           offset1, offset2, is_normalize, is_preprocess=True)
        data = gesture_dataset_shrec17.Dataset(root_dir, is_full)
        (x_train_all, y_train), (x_test_all, y_test) = data.load_data_with_feature(do_pca, pca_dim, M, max_dist_factor,
                                                                           offset1, offset2, is_normalize, is_preprocess=True)
        print 'elapsed time: {} s.'.format(time.time() - start_time)
        start_time = time.time()
        model = get_three_branches_model(data.input_dim, data.class_num, lr, is_bid)
    elif data == 6:  # preload data, batch generator with real-time feature extraction
        data = gesture_dataset.Dataset(root_dir, is_full)
        (x_train, y_train), (x_test, y_test) = data.load_data(test_id, is_preprocess=False)
        x_train_all = [x_train]
        _, (x_test_all, y_test) = data.load_data_with_feature(test_id, do_pca, pca_dim, M,
                                                                                   max_dist_factor,
                                                                                   offset1, offset2, is_normalize,
                                                                                   is_preprocess=True)
        print 'elapsed time: {} s.'.format(time.time() - start_time)
        start_time = time.time()
        model = get_three_branches_model_new(data.input_dim, data.class_num, lr, is_bid)
    elif data == 7:
        if dataset == 'SHREC17':
            data = gesture_dataset_shrec17.Dataset(root_dir, is_full)
            (x_train_all, y_train), (x_test_all, y_test) = data.load_data_with_vae_feature(do_pca, pca_dim, M, max_dist_factor,
                                                                           offset1, offset2, is_normalize, is_preprocess=True)
        else:
            data = gesture_dataset.Dataset(root_dir, is_full)
            (x_train_all, y_train), (x_test_all, y_test) = data.load_data_with_vae_feature(test_id, do_pca, pca_dim, M, max_dist_factor,
                                                                           offset1, offset2, is_normalize, is_preprocess=True)
                
        print 'elapsed time: {} s.'.format(time.time() - start_time)
        start_time = time.time()
        model = get_three_branches_model(data.input_dim, data.class_num, lr, is_bid)        
    data_gen = GestureDataGenerator(test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize)
    print 'elapsed time: {} s.'.format(time.time() - start_time)
    print 'Finish loading data!'
    print 'test_id: {}'.format(test_id)
    #model = get_baseline_model(data.input_dim, data.class_num, lr)
    # if mode == 0:
    #     rnn_model.train_model(model, x_train, y_train, x_test, y_test,
    #                         data.class_num, batch_size, num_epoch, save_dir)
    # elif mode == 1:
    #     rnn_model.test_model(model, x_test, y_test,
    #                         data.class_num, batch_size, save_dir)

    # model = get_two_branches_model(data.input_dim, data.class_num, lr)
    # model = get_three_branches_model(data.input_dim, data.class_num, lr)

    if mode == 0:
        rnn_model.train_model(model, x_train_all, y_train, x_test_all, y_test,
                              data.class_num, batch_size, num_epoch, save_postfix, MAX_LEN, save_best_only, test_id, data_gen)
        # rnn_model.train_model_online_feature(model, x_train_all, y_train, x_test_all, y_test,
        #                      data.class_num, batch_size, num_epoch, save_postfix, MAX_LEN, save_best_only, test_id, data_gen)
        # rnn_model.train_model2(model, x_train, x_train2, y_train, x_test, x_test2, y_test,
        #                     data.class_num, batch_size, num_epoch, save_postfix)
    elif mode == 1:
        rnn_model.test_model(model, x_test_all, y_test,
                            data.class_num, batch_size, save_postfix, MAX_LEN, test_id)


if __name__ == '__main__':
    # model = get_three_branches_model((100,30), 14, 0.001)
    # rnn_model.VisualizeModel(model, '.')
    main(*parse_arguments())
