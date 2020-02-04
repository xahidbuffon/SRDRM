#!/usr/bin/env python
"""
# > Script for training 4x generative SISR models on USR-248 data 
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
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # less logs
# local libs
from utils.plot_utils import save_val_samples
from utils.data_utils import dataLoaderUSR, deprocess
from utils.loss_utils import perceptual_distance, total_gen_loss
#############################################################################
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

###################################################################################
model_name = "srdrm" # ["res_sr", "image_sr", "sr_cnn", "denoise_sr", "srdrm"]
if (model_name.lower()=="res_sr"):
    from nets.gen_models import ResNetSR
    model_loader = ResNetSR(lr_shape, hr_shape, SCALE=4)
elif (model_name.lower()=="image_sr"):
    from nets.gen_models import ImageSR
    model_loader = ImageSR(lr_shape, hr_shape, SCALE=4)
elif (model_name.lower()=="sr_cnn"):
    from nets.gen_models import SRCNN
    model_loader = SRCNN(lr_shape, hr_shape, SCALE=4)
elif (model_name.lower()=="denoise_sr"):
    from nets.gen_models import DSRCNN
    model_loader = DSRCNN(lr_shape, hr_shape, SCALE=4)
else:
    print ("Using default model: srdrm")
    from nets.gen_models import SRDRM_gen
    model_loader = SRDRM_gen(lr_shape, hr_shape, SCALE=4)
# initialize the model
model = model_loader.create_model()
#print (model.summary())
# checkpoint directory
checkpoint_dir = os.path.join("checkpoints/", dataset_name, model_name)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
## sample directory
samples_dir = os.path.join("images/", dataset_name, model_name)
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

#####################################################################
# compile model

optimizer_ = Adam(0.0002, 0.5)
if (model_name.lower()=="srdrm"): 
    model.compile(optimizer=optimizer_, loss=total_gen_loss)
else:
    model.compile(optimizer=optimizer_, loss='mse')
print ("\nTraining: {0} with {1} data".format(model_name, dataset_name))
## training pipeline
step, epoch = 0, 0; start_time = datetime.datetime.now()
while (step <= num_step):
    for i, (imgs_lr, imgs_hr) in enumerate(data_loader.load_batch(batch_size)):
        # train the generator
        loss_i = model.train_on_batch(imgs_lr, imgs_hr)
        # increment step, and show the progress 
        step += 1; elapsed_time = datetime.datetime.now() - start_time
        if (step%50==0):
            print ("[Epoch %d: batch %d/%d] [loss_i: %f]" 
                               %(epoch, i+1, steps_per_epoch, loss_i))
        ## validate and save generated samples at regular intervals 
        if (step % sample_interval==0):
            imgs_lr, imgs_hr = data_loader.load_val_data(batch_size=2)
            fake_hr = model.predict(imgs_lr)
            gen_imgs = np.concatenate([deprocess(fake_hr), deprocess(imgs_hr)])
            save_val_samples(samples_dir, gen_imgs, step)
    # increment epoch, save model at regular intervals 
    epoch += 1
    ## save model and weights
    if (epoch%ckpt_interval==0):
        ckpt_name = os.path.join(checkpoint_dir, ("model_%d" %epoch))
        with open(ckpt_name+"_.json", "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(ckpt_name+"_.h5")
        print("\nSaved trained model in {0}\n".format(checkpoint_dir))

