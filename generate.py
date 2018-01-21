import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
import time
from tensorflow.python.platform import gfile
from net import *

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('input')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
args = parser.parse_args()

img = np.asarray(Image.open(args.input).convert('RGB'), dtype=np.float32)
input_ = img.reshape((1,) + img.shape)

if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'
with tf.device(device_):
    transform = FastStyleNet()
    image =  tf.placeholder(tf.float32, [1, input_.shape[1], input_.shape[2], 3])
    output = transform(image)
    saver = tf.train.Saver()
s_time = time.time()

with tf.Session() as sess: #config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    # Restore variables from disk.
    saver.restore(sess, args.model + '.ckpt')#"/tmp/model.ckpt")
    print("Model restored.")
    out = sess.run(output, feed_dict={image: input_})

print('time: {} sec'.format(time.time() - s_time))
out = out.reshape((out.shape[1:]))
im = Image.fromarray(np.uint8(out))
im.save(args.out)
