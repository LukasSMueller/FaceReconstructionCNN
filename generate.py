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
extensionsToCheck = ['.jpg', '.png']

# Check if output directory already exists
out_path = './' + args.out
if not os.path.exists(out_path):
    os.makedirs(out_path)

base, ext = os.path.splitext(args.input)
if ext == '.jpg' or ext == '.png':
#if any(ext in )
    img = np.asarray(Image.open(args.input).convert('RGB'), dtype=np.float32)
    input_ = img.reshape((1,) + img.shape)
    n_imgs = 1
    cut = base.rfind('/')
    fname = base[cut+1:]
else:
    fpath_inp = os.listdir(args.input)
    input_size = fpath_inp
    #inputpaths = []
    inputpaths = []
    fname = []
    for fn in fpath_inp:
        base, ext = os.path.splitext(fn)
        if ext == '.jpg' or ext == '.png':
            imagepath = os.path.join(args.input, fn)
            inputpaths.append(imagepath)
            fname.append(base)
    n_imgs = len(inputpaths)
    img = np.zeros((n_imgs, 112, 112, 3), dtype=np.float32)
    for i in range(n_imgs):
        img[i] = np.asarray(Image.open(inputpaths[i]).convert('RGB').resize((112, 112)), np.float32)
    input_ = img

if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'
with tf.device(device_):
    transform = FastStyleNet()
    image =  tf.placeholder(tf.float32, [n_imgs, input_.shape[1], input_.shape[2], 3])
    output = transform(image)
    saver = tf.train.Saver()
s_time = time.time()

with tf.Session() as sess: #config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    # Restore variables from disk.
    saver.restore(sess, args.model + '.ckpt')#"/tmp/model.ckpt")
    print("Model restored.")
    out = sess.run(output, feed_dict={image: input_})

print('time: {} sec'.format(time.time() - s_time))
print(out.shape)
if n_imgs == 1:
    out = out.reshape((out.shape[1:]))
    im = Image.fromarray(np.uint8(out))
    im.save(out_path + '/' + fname + '_rec.jpg')
else:
    for i in range(n_imgs):
        im = Image.fromarray(np.uint8(out[i]))
        im.save(out_path + '/' + fname[i] + '_rec.jpg')
print('Saved reconstructed images to {}'.format(out_path))
