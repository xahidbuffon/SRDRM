#!/usr/bin/env python
"""
# > Proposed SRDRM-GAN model 
#    - Paper: https://arxiv.org/pdf/x.y.pdf
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
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.models import Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Activation, Add
from keras.layers import Input, Dense
import keras.backend as K

class SRDRM_model():
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        # Input shapes
        self.SCALE = SCALE # SCALE = 2/4/8
        self.lr_shape, self.hr_shape = lr_shape, hr_shape
        self.lr_width, self.lr_height, self.channels = lr_shape
        self.hr_width, self.hr_height, _ = hr_shape
        # Number of residual blocks in the generator
        self.n_residual_blocks = 8
        optimizer = Adam(0.0002, 0.5)
        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg19()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (30, 40, 1)
        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # Build the generator
        self.generator = self.build_generator()
        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)
        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)
        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)
        self.combined = Model([img_lr, img_hr], [validity, fake_features, fake_hr])
        self.combined.compile(loss=['binary_crossentropy', 'mse', self.total_gen_loss], 
                              loss_weights=[1e-3, 1, 1e-3], optimizer=optimizer)

    def build_vgg19(self):
        # features of pre-trained VGG19 model at the third block
        vgg = VGG19(weights="imagenet")
        # Make trainable as False
        vgg.trainable = False
        for l in vgg.layers:
            l.trainable = False
        vgg.outputs = [vgg.get_layer('block5_conv4').output]
        img = Input(shape=self.hr_shape)
        img_features = vgg(img)
        return Model(img, img_features)

    def total_gen_loss(self, org_content, gen_content):
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

    def build_generator(self):
        # generator model
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.5)(d)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.5)(d)
            d = Add()([d, layer_input])
            return d
        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u
        def res_mult_2x(layer_input):
            l1 = Conv2D(64, kernel_size=4, strides=1, padding='same')(layer_input)
            l1 = Activation('relu')(l1)
            #r = l1
            # Propogate through residual blocks
            r = residual_block(l1, self.gf)
            for _ in range(self.n_residual_blocks - 1):
                r = residual_block(r, self.gf)
            # Post-residual block
            l2 = Conv2D(64, kernel_size=4, strides=1, padding='same')(r)
            l2 = BatchNormalization(momentum=0.8)(l2)
            l2 = Add()([l2, l1])
            # Upsampling
            layer_2x = deconv2d(l2)
            return layer_2x
        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)
        # DRM blocks
        o1 = res_mult_2x(img_lr)
        o2 = o1 if self.SCALE<4 else res_mult_2x(o1)
        o3 = o2 if self.SCALE<8 else res_mult_2x(o2)
        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(o3)
        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        # discriminator model
        def d_block(layer_input, filters, bn=True, strides=1):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn: d = BatchNormalization(momentum=0.45)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d
        # Input img
        d0 = Input(shape=self.hr_shape)
        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)
        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        return Model(d0, validity)



