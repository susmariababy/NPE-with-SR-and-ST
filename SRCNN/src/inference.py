from __future__ import print_function

import sys
sys.path.append('/usr/lib/python2.7')        # smb

import cv2

import os
import numpy as np
import cPickle as pickle
import timeit
import time

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def main(inputFile, outputFile, gpu = 0, color = 'rgb', arch='seranet_v1'):

    filepath = os.path.dirname(os.path.realpath(__file__))

    #DEBUG
    #inputFile = os.path.join(filepath, '../assets/compare/4/photo4_xinput.jpg')
    #outputFile = os.path.join(filepath, '../assets/compare/4/seranet_v1.jpg')

    input_file_path = inputFile
    if not os.path.exists(input_file_path):
        raise ValueError('input file ', os.path.dirname(input_file_path), ' not exist')

    if outputFile == None:
        file_name_with_ext = os.path.basename(inputFile) # returns filename from path
        filename_wo_ext, ext = os.path.splitext(file_name_with_ext)
        output_file_path = os.path.join(os.path.dirname(inputFile), filename_wo_ext + '-seranet.jpg')
        conventional_file_path = os.path.join(os.path.dirname(inputFile), filename_wo_ext + '-conventional.jpg')
    else:
        file_name_with_ext = os.path.basename(outputFile) # returns filename from path
        filename_wo_ext, ext = os.path.splitext(file_name_with_ext)
        output_file_path = outputFile
        conventional_file_path = os.path.join(os.path.dirname(outputFile), filename_wo_ext + '-conventional.jpg')
        output_file_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_file_dir):
            os.mkdir(output_file_dir)
            print('output file directory ', output_file_dir, ' not exist, created automatically')

    if gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu >= 0 else np

    if color == 'yonly':
        inout_ch = 1
    elif color == 'rgb':
        inout_ch = 3
    else:
        raise ValueError('Invalid color training scheme')

    # Prepare model
    print('prepare model')
    if arch == 'basic_cnn_tail':
        import arch.basic_cnn_tail as model_arch
        model = model_arch.basic_cnn_tail(inout_ch=inout_ch)

    elif arch == 'basic_cnn_middle':
        import arch.basic_cnn_middle as model_arch
        model = model_arch.basic_cnn_middle(inout_ch=inout_ch)
    elif arch == 'basic_cnn_head':
        import arch.basic_cnn_head as model_arch
        model = model_arch.basic_cnn_head(inout_ch=inout_ch)
    elif arch == 'basic_cnn_small':
        import arch.basic_cnn_small as model_arch
        model = model_arch.basic_cnn_small(inout_ch=inout_ch)
    elif arch == 'seranet':
        import arch.seranet_split as model_arch
        model = model_arch.seranet_split(inout_ch=inout_ch)
    elif arch == 'seranet_v1':
        import arch.seranet_v1 as model_arch
        model = model_arch.seranet_v1(inout_ch=inout_ch)

    else:
        raise ValueError('Invalid architecture name')
    arch_folder = model_arch.arch_folder
    # Directory/File setting for output
    output_folder = os.path.join(arch_folder, color, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #os.chdir(output_folder)
    inference_log_file_name = 'inference.log'
    inference_log_file = open(os.path.join(output_folder, inference_log_file_name), 'w')

    """ Model setup """
    print('setup model')
    model_load_path = os.path.join(arch_folder, color, 'training_process', 'my.model')
    serializers.load_npz(model_load_path, model)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
    model.train = False

    """ Load data """
    print('loading data')
    input_img = cv2.imread(input_file_path, cv2.IMREAD_COLOR)

    print('upscaling to ', output_file_path)
    if color == 'rgb':
        input_img = np.transpose(input_img[:, :, :], (2, 0, 1))
        input_img = input_img / 255.0  # Must be handled as float
        input_img = input_img.reshape((1, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
        x_data = model.preprocess_x(input_img)
        x = Variable(xp.asarray(x_data), volatile='on')
        output_img = model(x)
        if (gpu >= 0):
            output_data = cuda.cupy.asnumpy(output_img.data)
        else:
            output_data = output_img.data
        output_img = output_data[0].transpose(1, 2, 0) * 255.

    elif color == 'yonly':
        ycc_input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCR_CB)
        y_input_img = np.transpose(ycc_input_img[:, :, 0:1], (2, 0, 1))
        y_input_img = y_input_img / 255.0  # Must be handled as float
        y_input_img = y_input_img.reshape((1, y_input_img.shape[0], y_input_img.shape[1], y_input_img.shape[2]))

        x_data = model.preprocess_x(y_input_img)
        x = Variable(xp.asarray(x_data), volatile='on')
        output_y_img = model(x)
        if (gpu >= 0):
            output_y_data = cuda.cupy.asnumpy(output_y_img.data)
        else:
            output_y_data = output_y_img.data

        input_image_height = input_img.shape[0]
        input_image_width = input_img.shape[1]
        output_image_height = 2 * input_image_height
        output_image_width = 2 * input_image_width
        scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height),
                                interpolation=cv2.INTER_LANCZOS4)
        ycc_scaled_input_img = cv2.cvtColor(scaled_input_img, cv2.COLOR_BGR2YCR_CB)
        ycc_scaled_input_img[:, :, 0:1] = output_y_data[0].transpose(1, 2, 0) * 255. # (width, height, ch)
        output_img = cv2.cvtColor(ycc_scaled_input_img, cv2.COLOR_YCR_CB2BGR)

    print('saved to ', output_file_path)
    cv2.imwrite(output_file_path, output_img)

if __name__ == '__main__':
    if (len(sys.argv) == 6):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif (len(sys.argv) == 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif (len(sys.argv) == 3):
        main(sys.argv[1], sys.argv[2])
