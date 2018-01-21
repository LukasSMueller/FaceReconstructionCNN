import numpy as np
import os, sys
import argparse
from PIL import Image
import tensorflow as tf
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
#parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
args = parser.parse_args()

data_dict = loadWeightsData('./vgg16.npy')
batchsize = args.batchsize

n_epoch = args.epoch
#lambda_tv = args.lambda_tv
#lambda_f = args.lambda_feat
#lambda_s = args.lambda_style
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

#    # style target feature
#    # compute gram maxtrix of style target
#    vgg_s = custom_Vgg16(target, data_dict=data_dict)
#    feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
#    gram_ = [gram_matrix(l) for l in feature_]

    # content target feature
    vgg_c = custom_Vgg16(target, data_dict=data_dict)
    feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

    # feature after transformation
    vgg = custom_Vgg16(outputs, data_dict=data_dict)
    feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

    # compute feature loss
    loss_f = tf.zeros(batchsize, tf.float32)
    for f, f_ in zip(feature, feature_):
        loss_f += tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

    loss = loss_f
    tf.summary.scalar("loss", loss[0])

lr = tf. placeholder(tf.float32)
    # optimizer
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)


    # merge all the summaries
    summary = tf.summary.merge_all()

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
    learning_rates = [1E-3, 1E-4, 1E-5]
    #train with 3 different learning rates
    for k in range(3):
        learning_rate = learning_rates[k]
        hparam = make_hparam_string(learning_rate)
        print('Starting training with %s' % hparam)
        writer = tf.summary.FileWriter(LOGDIR + hparam)
        #writer.add_graph(sess.graph)
        for epoch in range(n_epoch):
            print ('epoch', epoch)
            imgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
            trgs = np.zeros((batchsize, 112, 112, 3), dtype=np.float32)
            for i in range(n_iter):
            #reading in all the images into the batch
                for j in range(batchsize):
                    p = imagepaths[i*batchsize + j]
                    q = targetpaths[i*batchsize + j]
                    imgs[j] = np.asarray(Image.open(p).convert('RGB').resize((112, 112)), np.float32)
                    trgs[j] = np.asarray(Image.open(q).convert('RGB').resize((112, 112)), np.float32)
                feed_dict = {inputs: imgs, target:trgs, lr: learning_rate}
                loss_, _, s_= sess.run([loss, train_step,summary,], feed_dict=feed_dict)
                writer.add_summary(s_, i)
                print('(epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter-1, loss_[0]))
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
