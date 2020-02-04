#!/usr/bin/env python
"""
# > Various modules for handling data 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from PIL import Image
from scipy import misc
from glob import glob

def deprocess(x):
    # [-1,1] -> [0, 1]
    return (x+1.0)*0.5

def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x/127.5)-1.0

def augment(a_img, b_img):
    """
       Augment images - a is distorted
    """
    # randomly interpolate
    a = random.random()
    #a_img = a_img*(1-a) + b_img*a
    # flip image left right
    if (random.random() < 0.25):
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    # flip image up down
    if (random.random() < 0.25):
        a_img = np.flipud(a_img)
        b_img = np.flipud(b_img) 
    return a_img, b_img

def getPaths(data_dir):
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)


def read_and_resize_pair(path_lr, path_hr, low_res=(60, 80), high_res=(480, 640)):
    img_lr = misc.imread(path_lr, mode='RGB').astype(np.float)  
    img_lr = misc.imresize(img_lr, low_res)
    img_hr = misc.imread(path_hr, mode='RGB').astype(np.float)
    img_hr = misc.imresize(img_hr, high_res)
    return img_lr, img_hr


class dataLoaderUSR():
    def __init__(self, DATA_PATH, SCALE=4):
        dataset_name = "USR-248"
        # SCALE = 2 (320, 240)  => (640,480)
        # SCALE = 4 (160, 120)   => (640,480)
        # SCALE = 8 (80, 60)    => (640,480)  
        self.SCALE = SCALE  #4x data if True (160, 120) => (640,480)
        self.lr_res_, self.low_res_folder_ = self.get_lr_info()
        train_dir = val_dir = os.path.join(DATA_PATH,  "train_val/")
        self.num_train, self.train_lr_paths, self.train_hr_paths = self.get_lr_hr_paths(train_dir)
        print ("Loaded {0} pairs of image-paths for training".format(self.num_train)) 
        self.num_val, self.val_lr_paths, self.val_hr_paths = self.get_lr_hr_paths(val_dir)

    def get_lr_info(self):
        if (self.SCALE==2):   lr_res, low_res_folder = (240, 320), "lr_2x/"
        elif (self.SCALE==8): lr_res, low_res_folder = (60, 80), "lr_8x/"
        else:    lr_res, low_res_folder = (120, 160), "lr_4x/"
        return lr_res, low_res_folder

    def get_lr_hr_paths(self, data_dir):
        lr_path = sorted(os.listdir(data_dir+self.low_res_folder_))      
        hr_path = sorted(os.listdir(data_dir+"hr/"))
        num_paths = min(len(lr_path), len(hr_path))
        all_lr_paths, all_hr_paths = [], []
        for f in lr_path[:num_paths]:
            all_lr_paths.append(os.path.join(data_dir+self.low_res_folder_, f))
            all_hr_paths.append(os.path.join(data_dir+"hr/", f))
        return num_paths, all_lr_paths, all_hr_paths

    def load_batch(self, batch_size=1, data_augment=True):
        self.n_batches = self.num_train//batch_size
        for i in range(self.n_batches-1):
            batch_lr = self.train_lr_paths[i*batch_size:(i+1)*batch_size]
            batch_hr = self.train_hr_paths[i*batch_size:(i+1)*batch_size]
            imgs_lr, imgs_hr = [], []
            for idx in range(len(batch_lr)): 
                img_lr, img_hr =  read_and_resize_pair(batch_lr[idx], batch_hr[idx], low_res=self.lr_res_)
                if (data_augment): img_lr, img_hr = augment(img_lr, img_hr)
                imgs_lr.append(img_lr)
                imgs_hr.append(img_hr)
            imgs_lr = preprocess(np.array(imgs_lr))
            imgs_hr = preprocess(np.array(imgs_hr))
            yield imgs_lr, imgs_hr

    def load_val_data(self, batch_size=2):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        paths_lr = [self.val_lr_paths[i] for i in idx]
        paths_hr = [self.val_hr_paths[i] for i in idx]
        imgs_lr, imgs_hr = [], []
        for idx in range(len(paths_hr)):
            img_lr, img_hr = read_and_resize_pair(paths_lr[idx], paths_hr[idx], low_res=self.lr_res_)
            imgs_lr.append(img_lr)
            imgs_hr.append(img_hr)
        imgs_lr = preprocess(np.array(imgs_lr))
        imgs_hr = preprocess(np.array(imgs_hr))
        return imgs_lr, imgs_hr


