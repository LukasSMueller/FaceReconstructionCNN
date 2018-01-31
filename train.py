import numpy as np
import random
import os, sys
import argparse
from PIL import Image

# Suppress some level of logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow import logging
logging.set_verbosity(logging.FATAL)

import time

from net import *
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
from custom_vgg16 import *


LOGDIR = "log_tb/"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='dataset/inputs', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--targetset', '-t', default='dataset/targets', type=str, required=True,
                    help='path to folder containing the target images')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', default='out', type=str,
                    help='output model file path without extension')
parser.add_argument('--lambda_tv', '-l_tv', default=10e-4, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', '-l_feat', default=1e0, type=float)
parser.add_argument('--lambda_style', '-l_style', default=1e1, type=float)
parser.add_argument('--epoch', '-e', default=150, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
parser.add_argument('--log', type=str,
                    help='name of the log entries')
args = parser.parse_args()

data_dict = loadWeightsData('./vgg16.npy')
batchsize = args.batchsize
n_epoch = args.epoch
output = args.output


# Read in all image paths from given dataset
fpath_inp = os.listdir(args.dataset)
fpath_trg = os.listdir(args.targetset)
# Size of total dataset and train, validation and test sets
n_data = int(len(fpath_inp))
n_train = int(np.floor(n_data*0.64))
n_val = int(np.ceil(n_data*0.16))
n_test = int(n_data*0.2)

# Handle exceptions
if batchsize > n_val:
    raise IOError('Entered batchsize is bigger than the validation set, please reduce batchsize')

#print(n_data)
indices = np.arange(n_data)
np.random.shuffle(indices)
#ind_train = indices[0:]
#ind_test = indices[]
iTrain, iVal, iTest = np.split(indices,[n_train, n_train+n_val])
all_images = fpath_inp
random.shuffle(fpath_inp)
inputpaths = []
targetpaths = []
# Sort paths of images into train, validation and test sets
# split into TRAINING{80%} [train (80%) &validation (20%)] and TEST DATA{20%}
# i.e. total data 10: 6 training, 2 validation, 2 test
for fn in fpath_inp:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
       imagepath = os.path.join(args.dataset,fn)
       inputpaths.append(imagepath)
       targetpath = os.path.join(args.targetset,fn)
       targetpaths.append(targetpath)
trainset, valset, testset = np.split(inputpaths,[n_train, n_train+n_val])
trainset_, valset_, testset_ = np.split(targetpaths,[n_train, n_train+n_val])
#n_data = len(imagepaths)
print ('Input images:', n_data)
print ('Training images:', n_train)
print ('Validation images:', n_val)
print ('Test images:', n_test)
n_iter = int(n_train / batchsize)
print (n_iter, 'iterations,', n_epoch, 'epochs')


if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'

with tf.device(device_):

    model = FastStyleNet()
    saver = tf.train.Saver(restore_sequentially=True)
    saver_def = saver.as_saver_def()

    inputs = tf.placeholder(tf.float32, shape=[batchsize, 112, 112, 3])
    #tf.summary.image('input' , inputs, 3)
    target = tf.placeholder(tf.float32, shape=[batchsize, 112, 112, 3])
    outputs = model(inputs)
with tf.name_scope("loss_network"):
    # initial input features
    with tf.name_scope("vgg16_on_input"):
        vgg_in = custom_Vgg16(inputs, data_dict=data_dict)
        feature_init = [vgg_in.conv1_2, vgg_in.conv2_2, vgg_in.conv3_3, vgg_in.conv4_3, vgg_in.conv5_3]
    # content target feature
    with tf.name_scope("vgg16_on_originals"):
        vgg_c = custom_Vgg16(target, data_dict=data_dict)
        feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]
    # feature after transformation
    with tf.name_scope("vgg16_on_output"):
        vgg = custom_Vgg16(outputs, data_dict=data_dict)
        feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]


    # compute initial loss of input data
    loss_i = tf.zeros(batchsize, tf.float32)
    for f_in, f_ in zip(feature_init, feature_):
        loss_i += tf.reduce_mean(tf.subtract(f_in, f_) ** 2, [1, 2, 3])

    megaloss = loss_i

    # compute feature loss
    loss_f = tf.zeros(batchsize, tf.float32)
    for f, f_ in zip(feature, feature_):
        loss_f += tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

    loss = loss_f

# optimizer
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

    # for calculating time
    s_time = time.time()

var_list={}

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    model_directory = './models/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # training
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("log_tb/")
    writer.add_graph(sess.graph)
    # Restore model if input is given
    if args.input:
        saver.restore(sess, args.input + '.ckpt')
        print ('restoring model ', args.input)
    # Set log dir to input or current date and time if no input specified
    if args.log:
        log_title = args.log
    else:
        log_title = time.strftime("%d/%m/%Y") + time.strftime("%H-%M-%S")


    writer = tf.summary.FileWriter(LOGDIR + log_title)

    for epoch in range(n_epoch):
        print ('epoch', epoch)
        imgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
        trgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
        imgs_val = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
        trgs_val = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
        loss_total = 0
        iLoss_total = 0
        s_total = 0
        # TRAINING
        for i in range(n_iter):
            #reading in all the images into the batch
            for j in range(batchsize):
                p = trainset[i*batchsize + j]
                q = trainset_[i*batchsize + j]
                imgs[j] = np.asarray(Image.open(p).convert('RGB').resize((112, 112)), np.float32)
                trgs[j] = np.asarray(Image.open(q).convert('RGB').resize((112, 112)), np.float32)
            feed_dict = {inputs: imgs, target:trgs}
            loss_, _, initial_loss = sess.run([loss, train_step, megaloss], feed_dict=feed_dict)
            #print TRAINING LOSS every i-th iteration
            if(i%2==0) and (i!=0):
                print('(Epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter-1, loss_[0]/initial_loss[0]))
            loss_total += loss_
            iLoss_total += initial_loss
            if(i%(n_iter-1)==0) and  (i!=0):
                # print VALIDATION LOSS at end of each epoch
                ind = random.sample(range(n_val), batchsize)
                for j in range(batchsize):
                    p = valset[ind[j]]
                    q = valset_[ind[j]]
                    imgs_val[j] = np.asarray(Image.open(p).convert('RGB').resize((112, 112)), np.float32)
                    trgs_val[j] = np.asarray(Image.open(q).convert('RGB').resize((112, 112)), np.float32)
                feed_dict = {inputs: imgs_val, target:trgs_val}
                loss_val_, initial_loss_val = sess.run([loss, megaloss], feed_dict=feed_dict)
                loss_val = np.sum(loss_val_) / np.sum(initial_loss_val)
                print('(Epoch {}) ... validation loss is...{}'.format(epoch, loss_val))
                summary = tf.Summary()
                summary.value.add(tag="Loss_Validation", simple_value=loss_val)
                writer.add_summary(summary, epoch)
        loss_total = np.sum(loss_total) / np.sum(iLoss_total)
        print('(Epoch {}) ... average training loss is...{}'.format(epoch, loss_total))
        summary = tf.Summary()
        summary.value.add(tag="Loss_Training", simple_value=loss_total)
        writer.add_summary(summary, epoch)

    #visualize one random output // NOT WORKING
    #rand_index = random.randint(0,n_train_data)
    #p = imagepaths[rand_index]
    #test_image = np.asarray(Image.open(p).convert('RGB').resize((112, 112)), np.float32)
    #test_image =tf. convert_to_tensor(test_image)
    #output = model(test_image) #MAKING PROBLEMS
    #tf.summary.image('Output' , output, 3)

    savepath = saver.save(sess, model_directory + args.output + '.ckpt')
    print('Saved the model to ', savepath)

    for var in tf.global_variables():
        var_list[var.name] = var.eval()
