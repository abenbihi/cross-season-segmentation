""" """
import os, glob
import re
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms
import h5py
import math

import sys
import time

from models import model_configs
from utils.segmentor_cmu import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
from utils.misc import rename_keys_to_match

# colmap output dir
MACHINE = 1
if MACHINE == 0:
    DATASET_DIR = '/home/abenbihi/ws/datasets/'
    WS_DIR = '/home/abenbihi/ws/'
    EXT_IMG_DIR = '/mnt/data_drive/dataset/Extended-CMU-Seasons/'
    #DATA_DIR = '/mnt/data_drive/dataset/CMU-Seasons/'
elif MACHINE == 1:
    #EXT_IMG_DIR = '/mnt/dataX/assia/Extended-CMU-Seasons/'
    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'

    # distorted img
    WS_DIR = '/home/gpu_user/assia/ws/'
    DATASET_DIR = '%s/datasets/'%WS_DIR
    EXT_IMG_DIR = '%s/Extended-CMU-Seasons/'%DATASET_DIR

    # undistorted img
    #EXT_IMG_DIR = '%s/Extended-CMU-Seasons-Undistorted/'%DATASET_DIR
else:
    print('Get you MTF MACHINE macro correct !')
    exit(1)

    
#META_DIR = '%s/life_saver/datasets/CMU-Seasons/meta/'%WS_DIR
META_DIR = "%s/datasets/pydata/pycmu/meta/surveys/"%WS_DIR
NETWORK_FILE = 'pth/from-paper/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19

XIAOLONG = (0==1)

def run_net(filenames_ims, filenames_segs):
    """Resize the img to 1024x1024 and segment patch by patch with overlap."""
    # network model
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)
    print('load model ' + NETWORK_FILE)
    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
            loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    # data proc
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, 2/3.)

    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    segmentor = Segmentor(net, net.n_classes, colorize_fcn =
            cityscapes.colorize_mask, n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for i, im_file in enumerate(filenames_ims):
        save_path = filenames_segs[i]
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)
        segmentor.run_and_save( im_file, save_path, save_folder,
                pre_sliding_crop_transform = pre_validation_transform,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=XIAOLONG)
        count += 1


def run_net_wo_resize(filenames_ims, filenames_segs):
    """Segment the whole img directly without resizing or croping (faster) and
    does not seem to lose accuracy on CMU-Seasons."""
    # network model
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)
    print('load model ' + NETWORK_FILE)
    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
            loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    # data proc
    input_transform = model_config.input_transform
    #pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, 2/3.)

    # encapsulate pytorch model in Segmentor class
    print("Class number: %d"%net.n_classes) # 19
    segmentor = Segmentor(net, net.n_classes, colorize_fcn =
            cityscapes.colorize_mask, n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for i, im_file in enumerate(filenames_ims):
        save_path = filenames_segs[i]
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)

        segmentor.run_and_save(im_file, save_path, save_folder,
                pre_sliding_crop_transform = None,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=XIAOLONG)
        count += 1
    
        #if count == 3:
        #    print("Early stop")
        #    break


def segment_slice(slice_id, save_folder):
    # output dir
    if not os.path.exists('%s/database/'%save_folder):
        os.makedirs('%s/database/'%save_folder)
    if not os.path.exists('%s/query/'%save_folder):
        os.makedirs('%s/query/'%save_folder)
    
    db_fn = glob.glob('%s/slice%d/database/*'%(EXT_IMG_DIR, slice_id))
    q_fn = glob.glob('%s/slice%d/query/*'%(EXT_IMG_DIR, slice_id))

    filenames_ims = db_fn + q_fn
    print(filenames_ims[0])
    print(filenames_ims[-1])

    filenames_segs = ['%s/%s.png' %(save_folder, 
        ('/'.join(l.split("/")[-2:]).split(".")[0])) 
        for l in filenames_ims]
    print(filenames_segs[0])
    print(filenames_segs[-1])
    
    #run_net(filenames_ims, filenames_segs)
    run_net_wo_resize(filenames_ims, filenames_segs)


def segment_survey(slice_id, cam_id, survey_id):
    # output dir
    #save_folder = 'res/ext_cmu_u/slice%d/'%(slice_id)
    if survey_id == -1:
        meta_fn = '%s/surveys/%d/pose/c%d_db.txt'%(META_DIR, slice_id, cam_id)
    else:
        meta_fn = '%s/surveys/%d/pose/c%d_%d.txt'%(META_DIR, slice_id, cam_id, survey_id)
    
    root_fn = np.loadtxt(meta_fn, dtype=str)[:,0]
    filenames_ims = ["%s/%s"%(EXT_IMG_DIR, l) for l in root_fn]
    print(filenames_ims[0])
    print(filenames_ims[-1])

    if XIAOLONG:
        filenames_segs = ['%s/col/%s.png' %(save_folder, 
            (l.split("/")[-1]).split(".")[0]) 
            for l in filenames_ims]
    else:
        filenames_segs = ['%s/%s.png' %(save_folder, 
            ('/'.join(l.split("/")[-2:]).split(".")[0])) 
            for l in filenames_ims]
    print(filenames_segs[0])
    print(filenames_segs[-1])
    
    run_net(filenames_ims, filenames_segs)
    #run_net_wo_resize(filenames_ims, filenames_segs)


if __name__ == '__main__':

    #for slice_id in range(22,23):
    #    if slice_id == 24:
    #        continue
    #    segment(slice_id)
    
    cam_id = 0

    for slice_id in [18]:
        if not XIAOLONG:
            save_folder = 'res/ext_cmu/slice%d/'%(slice_id) # distorted img
            if not os.path.exists('%s/query/'%save_folder):
                os.makedirs('%s/query/'%save_folder)
            if not os.path.exists('%s/database/'%save_folder):
                os.makedirs('%s/database/'%save_folder)

        # output dir
        if XIAOLONG:
            save_folder = 'res/ext_cmu_u/slice%d/'%(slice_id) # undistorted img
            if not os.path.exists('%s/col'%save_folder):
                os.makedirs('%s/col'%save_folder)
            if not os.path.exists('%s/lab'%save_folder):
                os.makedirs('%s/lab'%save_folder)
            if not os.path.exists('%s/prob'%save_folder):
                os.makedirs('%s/prob'%save_folder)

            for class_id in range(NUM_CLASS):
                if not os.path.exists('%s/prob/class_%d'%(save_folder, class_id)):
                    os.makedirs('%s/prob/class_%d'%(save_folder, class_id))
                #if not os.path.exists('%s/lab/class_%d'%(save_folder, class_id))

        segment_slice(slice_id, save_folder)
        #for survey_id in [-1, 0]:
        #    segment_survey(slice_id, cam_id, survey_id)

#
#
#    # get all file names
#    meta_fn = '%s/surveys/%d/fn/c%d_%d.txt'%(META_DIR, slice_id, cam_id,
#            survey_id)
#
#    filenames_ims = ['%s/slice%d/query/%s'%(EXT_IMG_DIR, slice_id, l) for l in
#            [ll.split("\n")[0] for ll in open(meta_fn, 'r').readlines()]]
#    filenames_segs = ['%s/col/%s.png'%(save_folder, l) for l
#            in [ll.split("\n")[0].split(".")[0] for ll in open(meta_fn, 'r').readlines()]]
#    
#    
#    #for i, l in enumerate(filenames_ims):
#    #    print(l)
#    #    input(filenames_segs[i])
#
#    # network model
#    print("Loading specified network from %s"%META_DIR)
#    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#    # Network and weight loading
#    model_config = model_configs.PspnetCityscapesConfig()
#    net = model_config.init_network().to(device)
#    print('load model ' + NETWORK_FILE)
#    state_dict = torch.load(NETWORK_FILE, map_location=lambda storage, 
#            loc: storage)
#    # needed since we slightly changed the structure of the network in pspnet
#    state_dict = rename_keys_to_match(state_dict)
#    net.load_state_dict(state_dict)
#    net.eval()
#
#
#    # data proc
#    input_transform = model_config.input_transform
#    pre_validation_transform = model_config.pre_validation_transform
#    # make sure crop size and stride same as during training
#    sliding_crop = joint_transforms.SlidingCropImageOnly(
#        713, 2/3.)
#
#
#    # encapsulate pytorch model in Segmentor class
#    print("Class number: %d"%net.n_classes) # 19
#    segmentor = Segmentor(
#            net,
#            net.n_classes,
#            colorize_fcn = cityscapes.colorize_mask,
#            n_slices_per_pass = 10)
#
#    # let's go
#    count = 1
#    t0 = time.time()
#    for im_file, save_path in zip(filenames_ims, filenames_segs):
#        tnow = time.time()
#        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
#            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
#        #print(save_path)
#        segmentor.run_and_save(
#            im_file,
#            save_path,
#            save_folder,
#            pre_sliding_crop_transform = pre_validation_transform,
#            sliding_crop = sliding_crop,
#            input_transform = input_transform,
#            skip_if_seg_exists = True,
#            use_gpu = True,
#            save_logits=True)
#        count += 1
#        #if count == 3:
#        #    break


