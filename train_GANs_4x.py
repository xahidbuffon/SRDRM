#!/usr/bin/env python
"""
# > Script for training 4x GAN-based SISR models on USR-248 data 
#    - Paper: https://arxiv.org/pdf/1909.09437.pdf
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
# keras libs
from keras.models import Model
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # less logs
# local libs
from utils.plot_utils import save_val_samples
from utils.data_utils import dataLoaderUSR, deprocess
#####################################################################
## dataset and image information
dataset_name = "USR_4x" # SCALE = 4
channels = 3
lr_width, lr_height = 160, 120   # low res
hr_width, hr_height = 640, 480 # high res (4x)
# input and output data
lr_shape = (lr_height, lr_width, channels)
hr_shape = (hr_height, hr_width, channels)
data_loader = dataLoaderUSR(DATA_PATH="/mnt/data1/ImageSR/USR-248/", SCALE=4)

# training parameters
num_epochs = 20
batch_size = 2
sample_interval = 500 # per step
ckpt_interval = 4 # per epoch
steps_per_epoch = (data_loader.num_train//batch_size)
num_step = num_epochs*steps_per_epoch
#####################################################################

# choose which model to run
model_name = "srdrm-gan" # options: ["srdrm-gan", "srgan", "esrgan", "edsrgan"]
if model_name.lower() == "srgan":
    from nets.SRGAN import SRGAN_model
    gan_model = SRGAN_model(lr_shape, hr_shape, SCALE=4)
elif (model_name.lower() =="esrgan"):
    from nets.ESRGAN import ESRGAN_model
    gan_model = ESRGAN_model(lr_shape, hr_shape, SCALE=4)
elif (model_name.lower() =="edsrgan"):
    from nets.EDSRGAN import EDSR_model
    gan_model = EDSR_model(lr_shape, hr_shape, SCALE=4)
else:
    print ("Using default model: SRDRM-GAN")
    from nets.SRDRM import SRDRM_model
    gan_model = SRDRM_model(lr_shape, hr_shape, SCALE=4)

# checkpoint directory
checkpoint_dir = os.path.join("checkpoints/", dataset_name, model_name)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
## sample directory
samples_dir = os.path.join("images/", dataset_name, model_name)
if not os.path.exists(samples_dir): os.makedirs(samples_dir)
#####################################################################

print ("\nGAN training: {0} with {1} data".format(model_name, dataset_name))
## ground-truths for adversarial loss
valid = np.ones((batch_size,) + gan_model.disc_patch)
fake = np.zeros((batch_size,) + gan_model.disc_patch)
step, epoch = 0, 0; start_time = datetime.datetime.now()
## training pipeline
while (step <= num_step):
    for i, (imgs_lr, imgs_hr) in enumerate(data_loader.load_batch(batch_size)):
        # train the discriminator
        fake_hr = gan_model.generator.predict(imgs_lr)
        d_loss_real = gan_model.discriminator.train_on_batch(imgs_hr, valid)
        d_loss_fake = gan_model.discriminator.train_on_batch(fake_hr, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # train the generators
        image_features = gan_model.vgg.predict(imgs_hr)
        if (model_name.lower()=="srdrm-gan"): 
            # custom loss function for SRDRM-GAN
            g_loss = gan_model.combined.train_on_batch([imgs_lr, imgs_hr], 
                                                       [valid, image_features, imgs_hr])
        else:
            g_loss = gan_model.combined.train_on_batch([imgs_lr, imgs_hr], 
                                                       [valid, image_features])
        # increment step, and show the progress 
        step += 1; elapsed_time = datetime.datetime.now() - start_time
        if (step%10==0):
            print ("[Epoch %d: batch %d/%d] [d_loss: %f] [g_loss: %03f]" 
                               %(epoch, i+1, steps_per_epoch, d_loss[0], g_loss[0]))
        ## validate and save generated samples at regular intervals 
        if (step % sample_interval==0):
            imgs_lr, imgs_hr = data_loader.load_val_data(batch_size=2)
            fake_hr = gan_model.generator.predict(imgs_lr)
            gen_imgs = np.concatenate([deprocess(fake_hr), deprocess(imgs_hr)])
            save_val_samples(samples_dir, gen_imgs, step)
    # increment epoch, save model at regular intervals 
    epoch += 1
    ## save model and weights
    if (epoch%ckpt_interval==0):
        ckpt_name = os.path.join(checkpoint_dir, ("model_%d" %epoch))
        with open(ckpt_name+"_.json", "w") as json_file:
            json_file.write(gan_model.generator.to_json())
        gan_model.generator.save_weights(ckpt_name+"_.h5")
        print("\nSaved trained model in {0}\n".format(checkpoint_dir))


