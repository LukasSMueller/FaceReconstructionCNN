import numpy as np
import random
import os, sys
import argparse
from PIL import Image

# Suppress some level of logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from custom_vgg16 import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Loss computation of reconstructed images')
parser.add_argument('--dataset', '-d', default='dataset/inputs', type=str,
                    help='path to folder containing the obfuscated images')
parser.add_argument('--testset', '-r', default='dataset/inputs', type=str,
                    help='path to folder containing the reconstructed images')
parser.add_argument('--targetset', '-t', default='dataset/targets', type=str, required=True,
                    help='path to folder containing the ground truth images')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
data_dict = loadWeightsData('./vgg16.npy')

# Read in all image paths from given dataset
fpath_inp = os.listdir(args.dataset)
fpath_test = os.listdir(args.testset)
fpath_trg = os.listdir(args.targetset)

# Store paths to images in list
imagepaths = []
inputpaths = []
targetpaths = []
# Sort paths of images into train, validation and test sets
# split into TRAINING{80%} [train (80%) &validation (20%)] and TEST DATA{20%}
# i.e. total data 10: 6 training, 2 validation, 2 test
for fn, fn_ in zip(fpath_test, fpath_trg):
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
       imagepath = os.path.join(args.testset,fn)
       imagepaths.append(imagepath)
       inputpath = os.path.join(args.dataset,fn_)
       inputpaths.append(inputpath)
       targetpath = os.path.join(args.targetset,fn_)
       targetpaths.append(targetpath)

# Read in all images into array
n_imgs = len(inputpaths)
print(n_imgs)
trgs = np.zeros((n_imgs, 112, 112, 3), dtype=np.float32)
inpts = np.zeros((n_imgs, 112, 112, 3), dtype=np.float32)
imgs = np.zeros((n_imgs, 112, 112, 3), dtype=np.float32)
for i in range(n_imgs):
    imgs[i] = np.asarray(Image.open(imagepaths[i]).convert('RGB').resize((112, 112)), np.float32)
    trgs[i] = np.asarray(Image.open(targetpaths[i]).convert('RGB').resize((112, 112)), np.float32)
    inpts[i] = np.asarray(Image.open(inputpaths[i]).convert('RGB').resize((112, 112)), np.float32)


if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'

with tf.device(device_):
    # Create placeholder for network input
    inputs = tf.placeholder(tf.float32, shape=[n_imgs, 112, 112, 3])
    target = tf.placeholder(tf.float32, shape=[n_imgs, 112, 112, 3])
    images = tf.placeholder(tf.float32, shape=[n_imgs, 112, 112, 3])

with tf.name_scope("loss_network"):
    # Target features
    with tf.name_scope("vgg16_on_originals"):
        vgg_c = custom_Vgg16(target, data_dict=data_dict)
        feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]
    # Reconstruction features
    with tf.name_scope("vgg16_on_output"):
        vgg = custom_Vgg16(images, data_dict=data_dict)
        feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
    # Initial features of obfuscated images
    with tf.name_scope("vgg16_on_input"):
        vgg_in = custom_Vgg16(inputs, data_dict=data_dict)
        feature_init = [vgg_in.conv1_2, vgg_in.conv2_2, vgg_in.conv3_3, vgg_in.conv4_3, vgg_in.conv5_3]

    # compute initial loss of input data
    loss_i = tf.zeros(n_imgs, tf.float32)
    for f_in, f_ in zip(feature_init, feature_):
        loss_i += tf.reduce_mean(tf.subtract(f_in, f_) ** 2, [1, 2, 3])

    # compute feature loss
    loss_f = tf.zeros(n_imgs, tf.float32)
    for f, f_ in zip(feature, feature_):
        loss_f += tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            feed_dict = {inputs: inpts, images: imgs, target:trgs}
            loss, loss_init = sess.run([loss_f, loss_i], feed_dict=feed_dict)

loss_rel = np.sum(loss)/np.sum(loss_init)
s_time = time.time()
print('Absolute loss is...{}, relative loss is...{}'.format(np.mean(loss), loss_rel))
