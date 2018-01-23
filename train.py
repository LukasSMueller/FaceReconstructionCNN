import numpy as np
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

#function to define the path where to store the values for the tensorboard
def make_hparam_string(learning_rate):
    return "lr_%.0E" % (learning_rate)

fpath_inp = os.listdir(args.dataset)
fpath_trg = os.listdir(args.targetset)
imagepaths = []
targetpaths = []
for fn, fn_ in zip(fpath_inp, fpath_trg):
    base, ext = os.path.splitext(fn)
    base_, ext_ = os.path.splitext(fn_)
    # TODO: check that input and target filenames are similar
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
    if ext_ == '.jpg' or ext_ == '.png':
        targetpath = os.path.join(args.targetset,fn_)
        targetpaths.append(targetpath)
n_data = len(imagepaths)
print ('num traning images:', n_data)
n_iter = int(n_data / batchsize)
print (n_iter, 'iterations,', n_epoch, 'epochs')

# style_ = np.asarray(Image.open(args.style_image).convert('RGB').resize((112,112)), dtype=np.float32)
# styles_ = [style_ for x in range(batchsize)]

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

    # initial input features
    vgg_in = custom_Vgg16(inputs, data_dict=data_dict)
    feature_init = [vgg_in.conv1_2, vgg_in.conv2_2, vgg_in.conv3_3, vgg_in.conv4_3, vgg_in.conv5_3]
    #feature_init = [vgg_in.conv1_2]

    # content target feature
    vgg_c = custom_Vgg16(target, data_dict=data_dict)
    feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]
    #feature_ = [vgg_c.conv1_2]

    # feature after transformation
    vgg = custom_Vgg16(outputs, data_dict=data_dict)
    feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
    #feature = [vgg.conv1_2]

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
    #loss = tf.reduce_mean(loss_f)
    #tf.summary.scalar("loss", loss[0])
    #tf.summary.scalar("loss", tf.reduce_mean(loss))

    # optimizer
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)


    # merge all the summaries
    #summary = tf.summary.merge_all()

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
    #hparam = make_hparam_string(args.lr)
    if args.log:
        log_title = args.log
    else:
        log_title = time.strftime("%d/%m/%Y") + time.strftime("%H-%M-%S")

    #print('Starting training with %s' % hparam)
    writer = tf.summary.FileWriter(LOGDIR + log_title)
    #writer.add_graph(sess.graph)

    ## Compute loss of obfuscated images
    #imgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
    #trgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
    #initLoss = 0
    #for i in range(n_iter):
    #    for j in range(batchsize):
    #        p = imagepaths[i*batchsize + j]
    #        q = targetpaths[i*batchsize + j]
    #        imgs[j] = np.asarray(Image.open(p).convert('RGB').resize((112, 112)), np.float32)
    #        trgs[j] = np.asarray(Image.open(q).convert('RGB').resize((112, 112)), np.float32)
    #    feed_dict = {inputs: imgs, target:trgs}
    #    inLoss = sess.run([megaloss], feed_dict=feed_dict)
    #    print(inLoss)
    #    initLoss += initLoss_
    #initLoss = np.sum(initLoss) / n_data
    #print("Initial average loss of obfuscated data: ", initialLoss)


    for epoch in range(n_epoch):
        print ('epoch', epoch)
        imgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
        trgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
        loss_total = 0
        iLoss_total = 0
        s_total = 0
        for i in range(n_iter):
            #reading in all the images into the batch
            for j in range(batchsize):
                p = imagepaths[i*batchsize + j]
                q = targetpaths[i*batchsize + j]
                imgs[j] = np.asarray(Image.open(p).convert('RGB').resize((112, 112)), np.float32)
                trgs[j] = np.asarray(Image.open(q).convert('RGB').resize((112, 112)), np.float32)
            feed_dict = {inputs: imgs, target:trgs}
            #loss_, _, s_ = sess.run([loss, train_step, summary,], feed_dict=feed_dict)
            loss_, _, initial_loss = sess.run([loss, train_step, megaloss], feed_dict=feed_dict)
            #print(initial_loss)
            #print(loss_)
            #writer.add_summary(s_, i)
            print('(epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter-1, loss_))
            loss_total += loss_
            iLoss_total += initial_loss
            #s_total += s_
        loss_total = (np.sum(loss_total) / n_data) / (np.sum(initial_loss)/n_data)
        print('(Epoch {}) ... training loss is...{}'.format(epoch, loss_total))
        summary = tf.Summary()
        summary.value.add(tag="Loss", simple_value=loss_total)
        writer.add_summary(summary, epoch)
        #writer.add_summary(s_, i)

    savepath = saver.save(sess, model_directory + args.output + '.ckpt')
    print('Saved the model to ', savepath)

    for var in tf.global_variables():
        var_list[var.name] = var.eval()


#Model = tf.Graph()
#with Model.as_default():
#    #with tf.device(device_):
#    inputs = tf.placeholder(tf.float32, shape=[1, 112, 112, 3], name='input')
#    # feed dictionary into Transform Net, "train=False" would save all values as constants.
#    transform = FastStyleNet(train=False, data_dict=var_list)
#    outputs = transform(inputs)
#    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#
#        save_path = './graphs/'
#        if not os.path.exists(save_path):
#            os.makedirs(save_path)
#        print('saving pb...')
#        tf.train.write_graph(sess.graph_def, save_path, args.output + '.pb', as_text=False)
