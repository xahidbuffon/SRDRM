from __future__ import print_function, division
import os
import sys
import numpy as np
# keras libs
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Add, Input, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, UpSampling2D

#############################################################
class BaseSRModel(object):
    def __init__(self, model_name, lr_shape, hr_shape, SCALE=4):
        # Base model to provide a standard interface
        self.model = None # type: Model
        self.lr_shape, self.hr_shape = lr_shape, hr_shape
        self.model_name = model_name  
        self.SCALE = SCALE # SCALE = 2/4/8
        

#############################################################
class ResNetSR(BaseSRModel):
    """  Derived (and simplified) from the "SRResNet" model of the SRGAN paper
         Photo-Realistic Single Image Super-Resolution Using a GAN
         https://arxiv.org/pdf/1609.04802.pdf 
    """
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        super(ResNetSR, self).__init__("ResNetSR", lr_shape, hr_shape, SCALE)
        self.n_residual_blocks = 8

    def create_model(self):
        # input layer followed by 3 conv layers
        init =  Input(shape=self.lr_shape)
        x0 = Convolution2D(64, (3, 3), activation='relu', padding='same')(init)
        x1 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=1)(x0)
        x2 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=1)(x1)
        # res layers
        x = self._residual_block(x2, 1)
        for i in range(self.n_residual_blocks - 1):
            x = self._residual_block(x, i + 2)
        # skip connect and up-scale
        x = Add()([x, x0])
        x = self._upscale_block(x)
        x = x if self.SCALE<4 else self._upscale_block(x)
        x = x if self.SCALE<8 else self._upscale_block(x)
        out = Convolution2D(3, (3, 3), activation="tanh", padding='same')(x)
        # return model
        return Model(init, out)

    def _residual_block(self, ip, id):
        init = ip
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = Convolution2D(64, (3, 3), activation='linear', padding='same', name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=False)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)
        x = Convolution2D(64, (3, 3), activation='linear', padding='same',name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=False)
        m = Add(name="sr_res_merge_" + str(id))([x, init])
        return m

    def _upscale_block(self, ip):
        init = ip
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = init._keras_shape[channel_dim]
        x = UpSampling2D()(init)
        x = Convolution2D(64, (3, 3), activation="relu", padding='same')(x)
        return x


#############################################################
class ImageSR(BaseSRModel):
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        super(ImageSR, self).__init__("Image SR", lr_shape, hr_shape, SCALE)
        self.f1, self.f2, self.f3 = 9, 1, 5
        self.gen_model = None # type: Model
        self.disc_model = None # type: Model
        self.type_scale_type = 'tanh'

    def create_model(self, load_weights=False):
        # Creates a model to be used to scale images of specific height and width.
        channel_axis = 1 if K.image_dim_ordering() == 'th' else -1
        init =  Input(shape=self.lr_shape)
        x = Convolution2D(64, (self.f1, self.f1), activation='relu', padding='same')(init)
        x = LeakyReLU(alpha=0.25)(x)
        x = Convolution2D(64, (self.f2, self.f2), activation='relu', padding='same')(x)
        x = LeakyReLU(alpha=0.25)(x)
        x = Convolution2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.25, name='disc_lr_1_1')(x)
        x = Convolution2D(64, (3, 3), padding='same',strides=1)(x)
        x = LeakyReLU(alpha=0.25, name='disc_lr_1_2')(x)
        x = Convolution2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.25, name='disc_lr_2_1')(x)
        x = Convolution2D(128, (3, 3), padding='same',strides=1)(x)
        x = LeakyReLU(alpha=0.25, name='disc_lr_2_2')(x)
        #x = Add()([x, x0])
        x = self.deconv2d(x)
        x = x if self.SCALE<4 else self.deconv2d(x)
        x = x if self.SCALE<8 else self.deconv2d(x)
        out = Convolution2D(3, (3, 3), activation="tanh", padding='same')(x)
        print (out)
        return Model(init, out)

    def deconv2d(self, layer_input):
        # Layers used during upsampling
        u = UpSampling2D(size=2)(layer_input)
        u = Convolution2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u


#############################################################
class DSRCNN(BaseSRModel):
    """  Derived (and simplified) from
         Deep Denoiseing Auto Encoder for Super Resolution (DSRCNN)
         https://arxiv.org/pdf/1606.08921.pdf 
    """
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        super(DSRCNN, self).__init__("Deep Denoise SR", lr_shape, hr_shape, SCALE)
        self.n1, self.n2, self.n3 = 64, 128, 256

    def create_model(self):
        # Perform check that model input shape is divisible by 4
        init = Input(shape=self.lr_shape)
        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1)
        x = MaxPooling2D((2, 2))(c1)
        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2)
        x = MaxPooling2D((2, 2))(c2)
        c3 = Convolution2D(self.n3, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D()(c3)
        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2_2)
        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)
        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1_2)
        m2 = Add()([c1, c1_2])
        x = UpSampling2D()(m2)
        x = x if self.SCALE<4 else UpSampling2D()(x)
        x = x if self.SCALE<8 else UpSampling2D()(x)
        decoded = Convolution2D(3, (5, 5), activation='tanh', border_mode='same')(x)
        print (decoded)
        return Model(init, decoded)


#############################################################
class SRCNN(BaseSRModel):
    """  Simplified implementation of the 
         Image Super-Resolution using CNNs (SRCNN)
         https://arxiv.org/pdf/1501.00092.pdf 
    """
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        super(SRCNN, self).__init__("Deep SRCNN", lr_shape, hr_shape, SCALE)
        self.f1, self.f2, self.f3 = 9, 1, 5
        self.n1, self.n2 = 64, 32

    def create_model(self):
        # input layer followed by 3 conv layers
        init =  Input(shape=self.lr_shape)
        x0 = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same')(init)
        x = Convolution2D(self.n2, (self.f2, self.f2), activation='relu', padding='same')(x0)
        # skip connect and up-scale
        #x = Add()([x, x0])
        x = self.deconv2d(x)
        x = x if self.SCALE<4 else self.deconv2d(x)
        x = x if self.SCALE<8 else self.deconv2d(x)
        out = Convolution2D(3, (self.f3, self.f3), activation='tanh', padding='same')(x)
        # return model
        return Model(init, out)

    def deconv2d(self, layer_input):
        # Layers used during upsampling
        u = UpSampling2D(size=2)(layer_input)
        u = Convolution2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u


#############################################################
class SRDRM_gen(BaseSRModel):
    """ Proposed SR model using Residual Multipliers
    """
    def __init__(self, lr_shape, hr_shape, SCALE=4):
        super(SRDRM_gen, self).__init__("SRDRM", lr_shape, hr_shape, SCALE)
        self.n_residual_blocks = 8
        self.gf = 64

    def residual_block(self, layer_input, filters):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.5)(d)
        d = Activation('relu')(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.5)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(self, layer_input):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

    def res_mult_2x(self, layer_input):
        l1 = Conv2D(64, kernel_size=4, strides=1, padding='same')(layer_input)
        l1 = Activation('relu')(l1)
        # Propogate through residual blocks
        r = self.residual_block(l1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = self.residual_block(r, self.gf)
        # Post-residual block
        l2 = Conv2D(64, kernel_size=4, strides=1, padding='same')(r)
        l2 = BatchNormalization(momentum=0.8)(l2)
        l2 = Add()([l2, l1])
        # Upsampling
        layer_2x = self.deconv2d(l2)
        return layer_2x

    def create_model(self):
        # Low resolution image input
        init = Input(shape=self.lr_shape)
        # Pre-residual block
        o1 = self.res_mult_2x(init)
        o2 = o1 if self.SCALE<4 else self.res_mult_2x(o1)
        o3 = o2 if self.SCALE<8 else self.res_mult_2x(o2) 
        # Generate high resolution output
        out = Conv2D(3, kernel_size=5, strides=1, padding='same', activation='tanh')(o3)
        return Model(init, out)

