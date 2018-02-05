import math
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
import time
from tensorflow.python.platform import gfile
from net_old import *

def getActivations(layer,stimuli):
    units = sess.run(layer, feed_dict={img: stimuli})
    #units = sess.run(layer,feed_dict={img:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

parser = argparse.ArgumentParser(description='Visualisation of neuron activations')
parser.add_argument('input')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
args = parser.parse_args()

# Check if output directory already exists
out_path = './' + args.out
if not os.path.exists(out_path):
    os.makedirs(out_path)

base, ext = os.path.splitext(args.input)
img = np.asarray(Image.open(args.input).convert('RGB'), dtype=np.float32)
img = img.reshape((1,) + img.shape)
n_imgs = 1
shape = img.shape
#cut = base.rfind('/')
#fname = base[cut+1:]

# INITIALIZE NETWORK
if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'
with tf.device(device_):
    net = FastStyleNet()
    feature = net.c1
    image =  tf.placeholder(tf.float32, [n_imgs, img.shape[1], img.shape[2], 3])
    #image =  tf.placeholder(tf.float32, [n_imgs, input_.shape[1], input_.shape[2], 3])
    output = net(image)
    saver = tf.train.Saver()
s_time = time.time()

with tf.Session() as sess: #config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    # Restore variables from disk.
    saver.restore(sess, args.model + '.ckpt')#"/tmp/model.ckpt")
    print("Model restored.")
    #getActivations('t_conv1_w:0',img)
    out = sess.run(output, feed_dict={image: img})
    toplot = out[5]
    print(toplot.shape)
    #print(toplot[0,:,:,0])
    #print(toplot[0,:,:,1])
    #print(feat)
    filters = toplot.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 10
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(toplot[0,:,:,i], interpolation="bicubic", cmap="gray")
    #getActivations(net.c1, img)
    plt.show()
