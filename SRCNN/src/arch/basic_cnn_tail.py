import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import os

import src.tools.image_processing as image_processing

""" Configuration """
file_path = os.path.dirname(os.path.realpath(__file__))
arch_folder = os.path.join(file_path, '../../data/arch/basic_cnn_small')

total_padding = 14


class basic_cnn_tail(Chain):
    """
    Basic CNN model.
    The network consists of Convolutional layer and leaky relu is used for activation
    Network structure is same with waifu2x
    """
    def __init__(self, inout_ch):
        super(basic_cnn_tail, self).__init__(
            conv1=L.Convolution2D(in_channels=inout_ch, out_channels=32, ksize=3, stride=1),
            conv2=L.Convolution2D(32, 32, 3, stride=1),
            conv3=L.Convolution2D(32, 64, 3, stride=1),
            conv4=L.Convolution2D(64, 64, 3, stride=1),
            conv5=L.Convolution2D(64, 128, 3, stride=1),
            conv6=L.Convolution2D(128, 128, 3, stride=1),
            conv7=L.Convolution2D(128, inout_ch, 3, stride=1),
        )
        self.train = True

    def __call__(self, x, t=None):
        self.clear()
        #x = Variable(x_data)  # x_data.astype(np.float32)

        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.clipped_relu(self.conv7(h), z=1.0)
        if self.train:
            self.loss = F.mean_squared_error(h, t)
            return self.loss
        else:
            return h

    def preprocess_x(self, x_data):
        """
        model specific preprocessing
        :param x_data:
        :return:
        """
        scaled_x = image_processing.nearest_neighbor_2x(x_data)
        return image_processing.image_padding(scaled_x, total_padding // 2)

    def clear(self):
        self.loss = None
        # self.accuracy = None

#    def forward(self, x, t):
#        self.clear()
#        #x = chainer.Variable(x_data)  # x_data.astype(np.float32)
#        #t = chainer.Variable(t_data)  # [Note]: x_data, t_data must be np.float32 type
#
#        #self.loss = F.huber_loss(h, t, delta= 1 / 255.)
#        self.loss = F.mean_squared_error(self(x), t)
#        # self.accuracy = F.accuracy(h, t)  # type inconpatible
#        return self.loss
#