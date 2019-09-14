

from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
from utils.misc import rename_keys_to_match

import os, sys, time, re, h5py, math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms


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
    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'
else:
    print('Get you MTF MACHINE macro correct !')
    exit(1)

    
META_DIR = '%s/life_saver/datasets/CMU-Seasons/meta/'%WS_DIR

NETWORK_FILE = 'pth/from-paper/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19


def run_net(filenames_ims, filenames_segs):

    # network model
    print("Loading specified network from %s"%META_DIR)
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
    segmentor = Segmentor(
            net,
            net.n_classes,
            colorize_fcn = cityscapes.colorize_mask,
            n_slices_per_pass = 10)

    # let's go
    count = 1
    t0 = time.time()
    for im_file, save_path in zip(filenames_ims, filenames_segs):
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)
        segmentor.run_and_save(
            im_file,
            save_path,
            '',
            pre_sliding_crop_transform = pre_validation_transform,
            sliding_crop = sliding_crop,
            input_transform = input_transform,
            skip_if_seg_exists = True,
            use_gpu = True,
            save_logits=False)
        count += 1
        #if count == 3:
        #    break


def segment_survey(slice_id, cam_id, survey_id, mode):
    # output dir
    if mode == 'database':
        save_folder = 'res/ext_cmu/%d/c%d_db/'%(slice_id, cam_id)
        meta_fn = '%s/surveys/%d/fn/c%d_db.txt'%(META_DIR, slice_id, cam_id)
    elif mode == 'query':
        save_folder = 'res/ext_cmu/%d/c%d_%d/'%(slice_id, cam_id, survey_id)
        meta_fn = '%s/surveys/%d/fn/c%d_%d.txt'%(META_DIR, slice_id, cam_id, survey_id)
    else:
        print("Error: wtf is this type of survey_id: ", type(survey_id))
        exit(1)


    meta = [ll.split("\n")[0] for ll in open(meta_fn, 'r').readlines()]
    filenames_ims = ['%s/slice%d/%s/%s'%(EXT_IMG_DIR, slice_id, mode, l) for l in meta]

    if not os.path.exists('%s/col'%save_folder):
        os.makedirs('%s/col'%save_folder)

    filenames_segs = ['%s/col/%s.png'%(save_folder, l) for l in meta]
    run_net(filenames_ims, filenames_segs)


def segment_for_vlad(data_id):
    
     
    meta_dir = '%s/retrieval/%d/'%(META_DIR, data_id)

    train_img_fn_v = np.loadtxt('%s/train_img.txt'%meta_dir, dtype=str)
    db_img_fn_v = np.loadtxt('%s/db_img.txt'%meta_dir, dtype=str)
    q_img_fn_v = np.loadtxt('%s/q_img.txt'%meta_dir, dtype=str)
    img_fn_v = np.hstack((train_img_fn_v, db_img_fn_v, q_img_fn_v))

    slices_present = np.unique(np.array([l.split("/")[0] for l in img_fn_v]))
    print(slices_present)
    
    for slice_id in slices_present:
        if not os.path.exists('res/ext_cmu/%s/query'%slice_id):
            os.makedirs('res/ext_cmu/%s/query'%slice_id)
        if not os.path.exists('res/ext_cmu/%s/database'%slice_id):
            os.makedirs('res/ext_cmu/%s/database'%slice_id)

    
    filenames_ims = ['%s/%s'%(EXT_IMG_DIR, l) for l in img_fn_v]
    filenames_segs = ['res/ext_cmu/%s.png'%l.split(".")[0] for l in img_fn_v]
    run_net(filenames_ims, filenames_segs)



if __name__ == '__main__':

    slice_id = 24
    cam_id = 0
    #for survey_id in range(1):
    #segment(slice_id, cam_id, None, 'database')
    
    survey_id = 0
    for survey_id in range(1,11):
        segment_survey(slice_id, cam_id, survey_id, 'query')
    
    #data_id = 1
    #segment_for_vlad(data_id)



