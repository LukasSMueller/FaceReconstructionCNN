import os
# Suppress some level of logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import logging
logging.set_verbosity(logging.FATAL)


def weight_variable(shape, name=None):
    # initialize weighted variables.
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)
def conv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    # set convolution layers.
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)
def batch_norm(x):
    assert isinstance(x, tf.Tensor)
    # reduce dimension 1, 2, 3, which would produce batch mean and batch variance.
    mean, var = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-5)
def relu(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.relu(x)
def deconv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None, mask_type=0):
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        kernel_h, kernel_w, c, kernel_n = W.get_shape().as_list()
        b, h, w, _ = x.get_shape().as_list()

        center_h = kernel_h // 2
        center_w = kernel_w // 2

        if mask_type != 0:
            mask = np.ones(
                (kernel_h, kernel_w, c, kernel_n), dtype=np.float32)

            mask[center_h, center_w+1:, :, :] = 0.
            mask[center_h+1:, :, :, :] = 0.

            if mask_type == 1:
                mask[center_h, center_w, :, :] = 0.

            W *= tf.constant(mask, dtype=tf.float32)

    return tf.nn.conv2d_transpose(x, W, [b, strides[1]*h, strides[1]*w, c], strides=strides, padding=p, name=name)
def max_pool_2x2(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#define a parametric relu
def parametric_relu(_x, alpha_):
    #with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    #    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
    #                   initializer=tf.constant_initializer(0.0),
    #                    dtype=tf.float32)

    pos = tf.nn.relu(_x)
    neg = alpha_ * (_x - abs(_x)) * 0.5

    return pos + neg

#define a conv layer with a name for better visualization in tensorboard
def conv_layer(x, W, alphas, strides=[1, 1, 1, 1], p='SAME', name="conv_layer"):
    # set convolution layers.
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        conv = tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)
        act = parametric_relu(conv, alphas)
        #act = tf.nn.relu(conv)
        mean, var = tf.nn.moments(act, axes=[1, 2, 3], keep_dims=True)
        return tf.nn.batch_normalization(act, mean, var, 0, 1, 1e-5)

#define a deconv layer for better visualization in tensorboard
def deconv_layer(x, W, alphas, strides=[1, 2, 2, 1], p='SAME', name="deconv_layer", mask=0):
    # set deconvolution layers.
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        deconv = deconv2d(x, W, strides=strides, name=name, mask_type=mask)
        #act = tf.nn.relu(deconv)
        act = parametric_relu(deconv, alphas)
        mean, var = tf.nn.moments(act, axes=[1, 2, 3], keep_dims=True)
        return tf.nn.batch_normalization(act, mean, var, 0, 1, 1e-5)

class ResidualBlock():
    def __init__(self, idx, ksize=3, filters=128, train=False, data_dict=None):
        if train:
            self.W1 = weight_variable([ksize, ksize, filters, filters], name='R'+str(idx)+'_conv1_w')
            self.W2 = weight_variable([ksize, ksize, filters, filters], name='R'+str(idx)+'_conv2_w')
        else:
            self.W1 = tf.constant(data_dict['R'+str(idx)+'_conv1_w:0'])
            self.W2 = tf.constant(data_dict['R'+str(idx)+'_conv2_w:0'])
    def __call__(self, x, idx, strides=[1, 1, 1, 1], name = "res_block"):
        with tf.name_scope(name):
            h = relu(batch_norm(conv2d(x, self.W1, strides, name='R'+str(idx)+'_conv1')))
            h = batch_norm(conv2d(h, self.W2, name='R'+ str(idx) + '_conv2'))
            return x + h


class FastStyleNet():
    def __init__(self, train=True, data_dict=None):
        print('initialize transform network...')
    #with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    #    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
    #                   initializer=tf.constant_initializer(0.0),
    #                    dtype=tf.float32)
        self.a1 = tf.Variable(tf.ones([1,112,112,32], name="alphas1"))
        self.a1 = self.a1*0.1
        self.a2 = tf.Variable(tf.ones([1,56,56,64], name="alphas2"))
        self.a2 = self.a2*0.1
        self.a3 = tf.Variable(tf.ones([1,28,28,128], name="alphas3"))
        self.a3 = self.a3*0.1
        self.a0 = tf.Variable(tf.ones([2,112,112,3], name="alphas3"))
        self.a0 = self.a0*0.1

        if train:
            self.c1 = weight_variable([9, 9, 3, 32], name='t_conv1_w')
            self.c2 = weight_variable([4, 4, 32, 64], name='t_conv2_w')
            self.c3 = weight_variable([4, 4, 64, 128], name='t_conv3_w')
            self.r1 = ResidualBlock(1, train=train)
            self.r2 = ResidualBlock(2, train=train)
            self.r3 = ResidualBlock(3, train=train)
            self.r4 = ResidualBlock(4, train=train)
            self.r5 = ResidualBlock(5, train=train)
            self.r1_1 = ResidualBlock(6, train=train, data_dict=data_dict, filters=64)
            self.r2_1 = ResidualBlock(7, train=train, data_dict=data_dict, filters=64)
            self.r3_1 = ResidualBlock(8, train=train, data_dict=data_dict, filters=64)
            self.r4_1 = ResidualBlock(9, train=train, data_dict=data_dict, filters=64)
            self.r5_1 = ResidualBlock(10, train=train, data_dict=data_dict, filters=64)
            self.r1_2 = ResidualBlock(11, train=train, data_dict=data_dict, filters=32)
            self.r2_2 = ResidualBlock(12, train=train, data_dict=data_dict, filters=32)
            self.r3_2 = ResidualBlock(13, train=train, data_dict=data_dict, filters=32)
            self.r4_2 = ResidualBlock(14, train=train, data_dict=data_dict, filters=32)
            self.r5_2 = ResidualBlock(15, train=train, data_dict=data_dict, filters=32)
            self.d1 = weight_variable([4, 4, 64, 128], name='t_dconv1_w')
            self.d2 = weight_variable([4, 4, 32, 64], name='t_dconv2_w')
            self.d3 = weight_variable([9, 9, 3, 32], name='t_dconv3_w')
        else:
            self.c1 = tf.constant(data_dict['t_conv1_w:0'])
            self.c2 = tf.constant(data_dict['t_conv2_w:0'])
            self.c3 = tf.constant(data_dict['t_conv3_w:0'])
            self.r1 = ResidualBlock(1, train=train, data_dict=data_dict)
            self.r2 = ResidualBlock(2, train=train, data_dict=data_dict)
            self.r3 = ResidualBlock(3, train=train, data_dict=data_dict)
            self.r4 = ResidualBlock(4, train=train, data_dict=data_dict)
            self.r5 = ResidualBlock(5, train=train, data_dict=data_dict)
            self.r1_1 = ResidualBlock(1, train=train, data_dict=data_dict)
            self.r2_1 = ResidualBlock(2, train=train, data_dict=data_dict)
            self.r3_1 = ResidualBlock(3, train=train, data_dict=data_dict)
            self.r4_1 = ResidualBlock(4, train=train, data_dict=data_dict)
            self.r5_1 = ResidualBlock(5, train=train, data_dict=data_dict)
            self.r1_2 = ResidualBlock(1, train=train, data_dict=data_dict)
            self.r2_2 = ResidualBlock(2, train=train, data_dict=data_dict)
            self.r3_2 = ResidualBlock(3, train=train, data_dict=data_dict)
            self.r4_2 = ResidualBlock(4, train=train, data_dict=data_dict)
            self.r5_2 = ResidualBlock(5, train=train, data_dict=data_dict)
            self.d1 = tf.constant(data_dict['t_dconv1_w:0'])
            self.d2 = tf.constant(data_dict['t_dconv2_w:0'])
            self.d3 = tf.constant(data_dict['t_dconv3_w:0'])
    def __call__(self, h):
        h1 = conv_layer(h, self.c1, self.a1)
        h2 = conv_layer(h1, self.c2, self.a2, strides=[1, 2, 2, 1])
        h3 = conv_layer(h2, self.c3, self.a3, strides=[1, 2, 2, 1])

        h4 = self.r1(h3, 1)
        h5 = self.r2(h4, 2)
        h6 = self.r3(h5, 3)
        h7 = self.r4(h6, 4)
        h8 = self.r5(h7, 5)

#h = batch_norm(relu(deconv2d(h, self.d1, strides=[1, 2, 2, 1], name='t_deconv1')))
#       h = batch_norm(relu(deconv2d(h, self.d2, strides=[1, 2, 2, 1], name='t_deconv2')))
        h9 = deconv_layer(h8, self.d1, self.a2, mask=0)
        h9 = self.r1_1(h9, 6)
        h9 = self.r2_1(h9, 7)
        h9 = self.r3_1(h9, 8)
        h9 = self.r4_1(h9, 9)
        h9 = self.r5_1(h9, 10)
        h10 = deconv_layer(h9, self.d2, self.a1, mask=0)
        h10 = self.r1_2(h10, 11)
        h10 = self.r2_2(h10, 12)
        h10 = self.r3_2(h10, 13)
        h10 = self.r4_2(h10, 14)
        h10 = self.r5_2(h10, 15)
        y = deconv2d(h10, self.d3, name='t_deconv3', mask_type=0)
        y = tf.multiply((tf.tanh(y) + 1), tf.constant(127.5, tf.float32, shape=y.get_shape()), name='output')
        output = [y, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]
        tf.summary.image('output', y,3)
        return output
