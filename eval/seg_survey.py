

from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
#from datasets import cityscapes, dataset_configs
#from utils.misc import check_mkdir, get_global_opts, rename_keys_to_match
from utils.misc import rename_keys_to_match

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

#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# colmap output dir
MACHINE = 1
if MACHINE == 0:
    DATASET_DIR = '/home/abenbihi/ws/datasets/'
    WS_DIR = '/home/abenbihi/ws/'
    EXT_IMG_DIR = '/mnt/data_drive/dataset/Extended-CMU-Seasons/'
    #DATA_DIR = '/mnt/data_drive/dataset/CMU-Seasons/'
elif MACHINE == 1:
    DATASET_DIR = '/home/gpu_user/assia/ws/datasets/'
    WS_DIR = '/home/gpu_user/assia/ws/'
    EXT_IMG_DIR = '/home/gpu_user/assia/ws/datasets/Extended-CMU-Seasons/'
    #EXT_IMG_DIR = '/mnt/dataX/assia/Extended-CMU-Seasons/'
    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'
else:
    print('Get you MTF MACHINE macro correct !')
    exit(1)

    
META_DIR = '%s/life_saver/datasets/CMU-Seasons/meta/'%WS_DIR

NETWORK_FILE = 'pth/from-paper/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19


def run_net(filenames_ims, filenames_segs):
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
    segmentor = Segmentor( net, net.n_classes, colorize_fcn =
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

        segmentor.run_and_save( im_file, save_path, '',
                pre_sliding_crop_transform = pre_validation_transform,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=False)
        count += 1 


## useful for xialong
#def segment(slice_id, cam_id, survey_id):
#
#    # output dir
#    save_folder = 'res/ext_cmu/slice%d/'%(slice_id, cam_id, survey_id)
#    if not os.path.exists('%s/col'%save_folder):
#        os.makedirs('%s/col'%save_folder)
#    if not os.path.exists('%s/lab'%save_folder):
#        os.makedirs('%s/lab'%save_folder)
#    if not os.path.exists('%s/prob'%save_folder):
#        os.makedirs('%s/prob'%save_folder)
#
#    for class_id in range(NUM_CLASS):
#        if not os.path.exists('%s/prob/class_%d'%(save_folder, class_id)):
#            os.makedirs('%s/prob/class_%d'%(save_folder, class_id))
#        #if not os.path.exists('%s/lab/class_%d'%(save_folder, class_id))
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



def segment(slice_id):

    # output dir
    save_folder = 'res/ext_cmu/slice%d/'%(slice_id)
    if not os.path.exists('%s/query/'%save_folder):
        os.makedirs('%s/query/'%save_folder)
    if not os.path.exists('%s/database/'%save_folder):
        os.makedirs('%s/database/'%save_folder)

    
    q_fn = glob.glob('%s/slice%d/query/*'%(EXT_IMG_DIR, slice_id))
    db_fn = glob.glob('%s/slice%d/database/*'%(EXT_IMG_DIR, slice_id))

    filenames_ims = q_fn + db_fn
    print(filenames_ims[0])
    print(filenames_ims[-1])

    filenames_segs = ['%s/%s.png' %(save_folder, 
        ('/'.join(l.split("/")[-2:]).split(".")[0])) 
        for l in filenames_ims]
    print(filenames_segs[0])
    print(filenames_segs[-1])

    
    run_net(filenames_ims, filenames_segs)


if __name__ == '__main__':

    #for slice_id in range(22,23):
    #    if slice_id == 24:
    #        continue
    #    segment(slice_id)
    slice_id = 24
    for slice_id in [2,8]:
        segment(slice_id)



