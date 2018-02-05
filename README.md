# FaceReconstructionCNN

Tensorflow implementation of a neural network used to reconstruct obfuscated human faces. Network implementation according to "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" adjusted to the application to face reconstruction.

## Prerequisites

For the loss computation the [implementation of a VGG16 model](https://github.com/machrisaa/tensorflow-vgg) is used. The data of a pretrained model can be obtained [here](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM).

## Neural Networks

Three slightly different networks have been implemented in the files net.py, net_old.py and deep_net.py. To switch between these nets the files for training (train.py) and generating (generate.py) have to be adapted. Therefore the line:
  from net import *
has to be changed to
  from deep_net import *
or
  from net_old import *

## Training the model

The model can be trained with the following command:
