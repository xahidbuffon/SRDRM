from __future__ import print_function, division
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
# keras libs
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
import keras.backend as K

class SRGAN_model():
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        # Input shapes
        self.SCALE = SCALE # SCALE = 2/4/8
        self.lr_shape, self.hr_shape = lr_shape, hr_shape
        self.lr_width, self.lr_height, self.channels = lr_shape
        self.hr_width, self.hr_height, _ = hr_shape
        # Number of residual blocks in the generator
        self.n_residual_blocks = 16
        optimizer = Adam(0.0002, 0.5)
        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
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
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)

    def build_vgg(self):
        # features of pre-trained VGG19 model at the third block
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=self.hr_shape)
        img_features = vgg(img)
        return Model(img, img_features)

    def build_generator(self):
        # generator model
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d
        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u
        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)
        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)
        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)
        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
        # Upsampling
        u1 = deconv2d(c2)
        u2 = u1 if self.SCALE<4 else deconv2d(u1)
        u3 = u2 if self.SCALE<8 else deconv2d(u2)
        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u3)
        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        # discriminator model
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
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

if __name__=="__main__":
    gan = SRGAN_model((80,60,3), (640,480,3))
    print (gan.generator.summary())


