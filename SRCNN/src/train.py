from __future__ import print_function

import sys
import cv2

import os
import numpy as np
import cPickle as pickle
import timeit
import time
from argparse import ArgumentParser

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from tools.prepare_data import load_data

if __name__ == '__main__':

    """ Pre setup """

    # Get params (Arguments)
    parser = ArgumentParser(description='SeRanet training')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--arch', '-a', default='basic_cnn_small',
                        help='model selection (basic_cnn_tail, basic_cnn_middle, basic_cnn_head, basic_cnn_small, '
                             'seranet_split, seranet_v1)')
    parser.add_argument('--batchsize', '-B', type=int, default=5, help='Learning minibatch size')
    #parser.add_argument('--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    parser.add_argument('--epoch', '-E', default=1000, type=int, help='Number of max epochs to learn')
    parser.add_argument('--color', '-c', default='rgb', help='training scheme for input/output color: (yonly, rgb)')
    parser.add_argument('--size', '-s', default=64, help='image crop size for training data, maximum 232')

    args = parser.parse_args()

    n_epoch = args.epoch           # #of training epoch
    batch_size = args.batchsize    # size of minibatch
    visualize_test_img_number = 5  # #of images to visualize for checking training performance
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.color == 'yonly':
        inout_ch = 1
    elif args.color == 'rgb':
        inout_ch = 3
    else:
        raise ValueError('Invalid color training scheme')

    # Prepare model
    print('prepare model')
    if args.arch == 'basic_cnn_tail':
        import arch.basic_cnn_tail as model_arch
        model = model_arch.basic_cnn_tail(inout_ch=inout_ch)

    elif args.arch == 'basic_cnn_middle':
        import arch.basic_cnn_middle as model_arch
        model = model_arch.basic_cnn_middle(inout_ch=inout_ch)
    elif args.arch == 'basic_cnn_head':
        import arch.basic_cnn_head as model_arch
        model = model_arch.basic_cnn_head(inout_ch=inout_ch)
    elif args.arch == 'basic_cnn_small':
        import arch.basic_cnn_small as model_arch
        model = model_arch.basic_cnn_small(inout_ch=inout_ch)
    elif args.arch == 'seranet_split':
        import arch.seranet_split as model_arch
        model = model_arch.seranet_split(inout_ch=inout_ch)
    elif args.arch == 'seranet_v1':
        import arch.seranet_v1 as model_arch
        model = model_arch.seranet_v1(inout_ch=inout_ch)

    else:
        raise ValueError('Invalid architecture name')
    arch_folder = model_arch.arch_folder
    # Directory/File setting for training log
    training_process_folder = os.path.join(arch_folder, args.color, 'training_process')

    if not os.path.exists(training_process_folder):
        os.makedirs(training_process_folder)
    os.chdir(training_process_folder)
    train_log_file_name = 'train.log'
    train_log_file = open(os.path.join(training_process_folder, train_log_file_name), 'w')
    #total_image_padding = 14 #24 #18 #14

    print("""-------- training parameter --------
GPU ID        : %d
archtecture   : %s
batch size    : %d
epoch         : %d
color scheme  : %s
size          : %d
------------------------------------
""" % (args.gpu, args.arch, args.batchsize, args.epoch, args.color, args.size))
    print("""-------- training parameter --------
GPU ID        : %d
archtecture   : %s
batch size    : %d
epoch         : %d
color scheme  : %s
size          : %d
------------------------------------
""" % (args.gpu, args.arch, args.batchsize, args.epoch, args.color, args.size), file=train_log_file)

    """ Load data """
    print('loading data')

    """
    Changing crop_size to smaller value makes training speed faster
    crop_size=30 for easy demo, 64 for default value, maximum 232
    """
    datasets = load_data(mode=args.color, crop_size=args.size)

    np_train_dataset, np_valid_dataset, np_test_dataset = datasets
    np_train_set_x, np_train_set_y = np_train_dataset
    np_valid_set_x, np_valid_set_y = np_valid_dataset
    np_test_set_x, np_test_set_y = np_test_dataset

    n_train = np_train_set_x.shape[0]
    n_valid = np_valid_set_x.shape[0]
    n_test = np_test_set_x.shape[0]

    """ Preprocess """
    #print('preprocess')
    epoch_start_time = timeit.default_timer()

    def normalize_image(np_array):
        np_array /= 255.
        np_array.astype(np.float32)

    normalize_image(np_train_set_x)
    normalize_image(np_valid_set_x)
    normalize_image(np_test_set_x)
    normalize_image(np_train_set_y)
    normalize_image(np_valid_set_y)
    normalize_image(np_test_set_y)

    epoch_end_time = timeit.default_timer()
    #print('preprocess time %i sec' % (end_time - start_time))
    #print('preprocess time %i sec' % (end_time - start_time), file=train_log_file)

    """ SHOW Test images (0~visualize_test_img_number) """
    for i in xrange(visualize_test_img_number):
        cv2.imwrite(os.path.join(training_process_folder, 'photo' + str(i) + '_xinput.jpg'),
                    np_test_set_x[i].transpose(1, 2, 0) * 255.)
        cv2.imwrite(os.path.join(training_process_folder, 'photo' + str(i) + '_original.jpg'),
                    np_test_set_y[i].transpose(1, 2, 0) * 255.)

    """ Setup GPU """
    """ Model, optimizer setup """
    print('setup model')
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=0.0001)
    # optimizer = optimizers.AdaDelta()
    # optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)


    """
    TRAINING
    Early stop method is used for training to avoid overfitting,
    Reference: https://github.com/lisa-lab/DeepLearningTutorials
    """
    print('training')

    patience = 30000
    patience_increase = 2
    improvement_threshold = 0.997  # 0.995

    validation_frequency = min(n_train, patience // 2) * 2

    best_validation_loss = np.inf
    iteration = 0
    best_iter = 0
    test_score = 0.
    done_looping = False

    training_start_time = timeit.default_timer()

    for epoch in xrange(1, n_epoch + 1):
        print('epoch: %d' % epoch)
        epoch_start_time = timeit.default_timer()
        perm = np.random.permutation(n_train)
        sum_loss = 0

        for i in xrange(0, n_train, batch_size):
            # start_iter_time = timeit.default_timer()
            iteration += 1
            if iteration % 100 == 0:
                print('training @ iter ', iteration)
            x_batch = np_train_set_x[perm[i: i + batch_size]].copy()
            y_batch = np_train_set_y[perm[i: i + batch_size]].copy()
            x_batch = model.preprocess_x(x_batch)
            # print('x_batch', x_batch.shape, x_batch.dtype)

            x = Variable(xp.asarray(x_batch))
            t = Variable(xp.asarray(y_batch))

            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * batch_size

            # end_iter_time = timeit.default_timer()
            # print("iter took: %f sec" % (end_iter_time - start_iter_time))

        print("train mean loss: %f" % (sum_loss / n_train))
        print("train mean loss: %f" % (sum_loss / n_train), file=train_log_file)

        # Validation
        sum_loss = 0
        for i in xrange(0, n_valid, batch_size):
            x_batch = np_valid_set_x[i: i + batch_size]
            y_batch = np_valid_set_y[i: i + batch_size]

            x_batch = model.preprocess_x(x_batch)

            x = Variable(xp.asarray(x_batch, dtype=xp.float32))
            t = Variable(xp.asarray(y_batch, dtype=xp.float32))

            sum_loss += float(model(x, t).data) * batch_size

        this_validation_loss = (sum_loss / n_valid)
        print("valid mean loss: %f" % this_validation_loss)
        print("valid mean loss: %f" % this_validation_loss, file=train_log_file)
        if this_validation_loss < best_validation_loss:
            if this_validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, iteration * patience_increase)
                print('update patience -> ', patience, ' iteration')

            best_validation_loss = this_validation_loss
            best_iter = iteration

            sum_loss = 0
            for i in xrange(0, n_test, batch_size):
                x_batch = np_test_set_x[i: i + batch_size]
                y_batch = np_test_set_y[i: i + batch_size]

                x_batch = model.preprocess_x(x_batch)

                x = Variable(xp.asarray(x_batch, dtype=xp.float32))
                t = Variable(xp.asarray(y_batch, dtype=xp.float32))

                sum_loss += float(model(x, t).data) * batch_size
            test_score = (sum_loss / n_test)
            print('  epoch %i, test cost of best model %f' %
                  (epoch, test_score))
            print('  epoch %i, test cost of best model %f' %
                  (epoch, test_score), file=train_log_file)

            # Save best model
            print('saving model')
            serializers.save_npz('my.model', model)
            serializers.save_npz('my.state', optimizer)

        if patience <= iteration:
            done_looping = True
            print('done_looping')
            break

        # Check test images
        if epoch // 10 == 0 or epoch % 10 == 0:
            model.train = False
            x_batch = np_test_set_x[0:5]
            x_batch = model.preprocess_x(x_batch)
            x = Variable(xp.asarray(x_batch, dtype=xp.float32))
            output = model(x)

            if (args.gpu >= 0):
                output_data = cuda.cupy.asnumpy(output.data)
            else:
                output_data = output.data

            for photo_id in xrange(visualize_test_img_number):
                cv2.imwrite(os.path.join(training_process_folder,
                                         'photo' + str(photo_id) + '_epoch' + str(epoch) + '.jpg'),
                            output_data[photo_id].transpose(1, 2, 0) * 255.)
            output = None  # It is important to release memory!
            model.train = True

        epoch_end_time = timeit.default_timer()
        print('epoch %i took %i sec' % (epoch, epoch_end_time - epoch_start_time))
        print('epoch %i took %i sec' % (epoch, epoch_end_time - epoch_start_time), file=train_log_file)
    training_end_time = timeit.default_timer()
    print('total training time took %i sec' % (training_end_time - training_start_time))
    print('total training time took %i sec' % (training_end_time - training_start_time), file=train_log_file)

