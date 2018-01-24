import os
# Suppress some level of logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt

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
def deconv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        _, _, c, _ = W.get_shape().as_list()
        b, h, w, _ = x.get_shape().as_list()
    return tf.nn.conv2d_transpose(x, W, [b, strides[1]*h, strides[1]*w, c], strides=strides, padding=p, name=name)
def max_pool_2x2(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#define a conv layer with a name for better visualization in tensorboard
def conv_layer(x, W, strides=[1, 1, 1, 1], p='SAME', name="conv_layer"):
    # set convolution layers.
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        conv = tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)
        act = tf.nn.relu(conv)
        mean, var = tf.nn.moments(act, axes=[1, 2, 3], keep_dims=True)
        return tf.nn.batch_normalization(act, mean, var, 0, 1, 1e-5)

#define a deconv layer for better visualization in tensorboard
def deconv_layer(x, W, strides=[1, 2, 2, 1], p='SAME', name="deconv_layer"):
    # set deconvolution layers.
    with tf.name_scope(name):
        assert isinstance(x, tf.Tensor)
        deconv = deconv2d(x, W, strides=strides, name=name)
        act = tf.nn.relu(deconv)
        mean, var = tf.nn.moments(act, axes=[1, 2, 3], keep_dims=True)
        return tf.nn.batch_normalization(act, mean, var, 0, 1, 1e-5)

class ResidualBlock():
    def __init__(self, idx, ksize=3, train=False, data_dict=None):
        if train:
            self.W1 = weight_variable([ksize, ksize, 128, 128], name='R'+str(idx)+'_conv1_w')
            self.W2 = weight_variable([ksize, ksize, 128, 128], name='R'+str(idx)+'_conv2_w')
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
        if train:
            self.c1 = weight_variable([9, 9, 3, 32], name='t_conv1_w')
            self.c2 = weight_variable([4, 4, 32, 64], name='t_conv2_w')
            self.c3 = weight_variable([4, 4, 64, 128], name='t_conv3_w')
            self.r1 = ResidualBlock(1, train=train)
            self.r2 = ResidualBlock(2, train=train)
            self.r3 = ResidualBlock(3, train=train)
            self.r4 = ResidualBlock(4, train=train)
            self.r5 = ResidualBlock(5, train=train)
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
            self.d1 = tf.constant(data_dict['t_dconv1_w:0'])
            self.d2 = tf.constant(data_dict['t_dconv2_w:0'])
            self.d3 = tf.constant(data_dict['t_dconv3_w:0'])
    def __call__(self, h):
        h = conv_layer(h, self.c1)
        h = conv_layer(h, self.c2, strides=[1, 2, 2, 1])
        h = conv_layer(h, self.c3, strides=[1, 2, 2, 1])

        h = self.r1(h, 1)
        h = self.r2(h, 2)
        h = self.r3(h, 3)
        h = self.r4(h, 4)
        h = self.r5(h, 5)

#h = batch_norm(relu(deconv2d(h, self.d1, strides=[1, 2, 2, 1], name='t_deconv1')))
#       h = batch_norm(relu(deconv2d(h, self.d2, strides=[1, 2, 2, 1], name='t_deconv2')))
        h = deconv_layer(h, self.d1)
        h = deconv_layer(h, self.d2)
        y = deconv2d(h, self.d3, name='t_deconv3')
        y = tf.multiply((tf.tanh(y) + 1), tf.constant(127.5, tf.float32, shape=y.get_shape()), name='output')
        tf.summary.image('output', y,3)
        return y
