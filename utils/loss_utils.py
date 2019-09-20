#!/usr/bin/env python
"""
# > Various modules for computing loss 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import print_function, division
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
# keras libs
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
import keras.backend as K

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def build_vgg19(hr_shape):
    # features of pre-trained VGG19 model at the third block
    vgg = VGG19(weights="imagenet")
    # Make trainable as False
    vgg.trainable = False
    for l in vgg.layers:
        l.trainable = False
    vgg.outputs = [vgg.get_layer('block5_conv4').output]
    img = Input(shape=hr_shape)
    img_features = vgg(img)
    return Model(img, img_features)

def total_gen_loss(org_content, gen_content):
    # mse loss
    mse_gen_loss = K.mean(K.square(org_content-gen_content))
    # perceptual loss
    y_true = (org_content+1.0)*127.5 # [-1,1] => [0,255]
    y_pred = (gen_content+1.0)*127.5 # [-1,1] => [0,255]
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    percep_loss = K.mean((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256))/255.0
    gen_total_err = 0.8*mse_gen_loss+0.2*percep_loss
    return gen_total_err




