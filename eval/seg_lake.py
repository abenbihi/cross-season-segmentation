

from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes
from utils.misc import rename_keys_to_match

import os, glob
import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import interp1d
import torch
import torchvision.transforms as standard_transforms
import time
from math import pi,cos,sin 
from pyquaternion import Quaternion

import cst


ROOT_DATA_DIR = '/mnt/lake/'
ROOT_SEG_DIR = '/home/gpu_user/assia/ws/datasets/lake/datasets/seg/'
NETWORK_FILE = 'pth/from-paper/CMU-CS-Vistas-CE.pth'
NUM_CLASS = 19


def run_net(filenames_ims, filenames_segs, filenames_mask):
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
        if filenames_mask is not None:
            mask_path = filenames_mask[i]
        else:
            mask_path = None
        tnow = time.time()
        print( "[%d/%d (%.1fs/%.1fs)] %s" % (count, len(filenames_ims), 
            tnow - t0, (tnow - t0) / count * len(filenames_ims), im_file))
        #print(save_path)

        #segmentor.run_and_save( im_file, save_path, '',
        #        pre_sliding_crop_transform = pre_validation_transform,
        #        sliding_crop = sliding_crop, input_transform = input_transform,
        #        skip_if_seg_exists = True, use_gpu = True, save_logits=False,
        #        mask_path=mask_path)

        segmentor.run_and_save( im_file, save_path, '',
                pre_sliding_crop_transform = None,
                sliding_crop = sliding_crop, input_transform = input_transform,
                skip_if_seg_exists = True, use_gpu = True, save_logits=False,
                mask_path=mask_path)
        count += 1 


def segment(survey_id, seq_start, seq_end, iter_):
    # output dir
    out_root_dir = 'res/%d'%survey_id

    img_dir = '%s/Dataset/20%d/%d/'%(ROOT_DATA_DIR, survey_id/10000, survey_id)
    mask_dir = '%s/%d/water/auto/'%(ROOT_SEG_DIR, survey_id)
    print(survey_id)
    seq_dir_l = sorted(glob.glob('%s/00*'%img_dir))

    filenames_ims, filenames_segs = [],[]
    filenames_mask = []

    for seq_dir in seq_dir_l:
        seq = int(os.path.basename(seq_dir))
        if int(seq)<seq_start:
            continue
        if int(seq) == seq_end:
            break 
        print('\n%04d'%seq)
        seq_img_fn_l = sorted(os.listdir(seq_dir))
        img_num = len(seq_img_fn_l)

        out_dir = '%s/%04d/lab/'%(out_root_dir, seq)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        #for i, img_root_fn in enumerate(seq_img_fn_l[img_start:img_end+1]):
        for i in range(0, img_num, iter_):
            img_root_fn = seq_img_fn_l[i]
            if i%100==0:
                print('%d  %04d  %d/%d'%(survey_id, seq, i, img_num))

            img_fn = '%s/%s'%(seq_dir, img_root_fn)
            out_fn = '%s/%s.png'%(out_dir, img_root_fn.split(".")[0])
            mask_fn = '%s/%04d/%s'%(mask_dir, seq, img_root_fn)
            
            # visual debug
            if (0==1):
                img = cv2.imread(img_fn)[:,:cst.W]
                mask = cv2.imread(mask_fn, cv2.IMREAD_UNCHANGED)
                print(mask.shape)
                print(img.shape)
                img[mask==1] = 0
                cv2.imshow('img', img)
                cv2.waitKey(0)

            #print('%s\n%s'%(img_fn, out_fn))
            print('mask_fn: %s'%mask_fn)
            filenames_ims.append(img_fn)
            filenames_segs.append(out_fn)
            filenames_mask.append(mask_fn)


    run_net(filenames_ims, filenames_segs, filenames_mask)

    

def get_good_pose(survey_id, seq_start, seq_end, iter_, survey_dir, pose_dir):
    # Load img list and timespamps
    survey_f = '%s/%s/image_auxilliary.csv'%(survey_dir, survey_id)
    img_aux_f = open(survey_f)
    new_names = ['t_sec','seq','x','y','theta','pan','tilt','fx','fy','cx','cy','width','height','omega','battery','rc']
    survey = pd.read_csv(img_aux_f,dtype=np.float64,comment='%',names=new_names)
    survey = survey[np.abs(survey['tilt']-0.199948)<1e-5] #delete the part before camera is ready 
    img_id_l = survey['seq'].values.astype(int) # img id list
    img_t_l = survey['t_sec'].values # img timestamps
    img_pan_l = survey['pan'].values # img camera pan
    
    
    # Load better poses than gps [(timestamps, pose)]
    optposes_fn = '%s/%s/optposes.csv'%(pose_dir, survey_id) #t,pose
    optposes_f = open(optposes_fn)
    opt_new_names = ['kf','t','x','y','theta','xg','yg','thetag']
    optposes = pd.read_csv(optposes_f,dtype=np.float64,comment='%',names=opt_new_names)
    pose_kf_l = optposes['kf'].values
    pose_t_l = optposes['t'].values
    pose_x_l = optposes['xg'].values
    pose_y_l = optposes['yg'].values
    pose_theta_l=optposes['thetag'].values
    
    # interpolation functions
    fx     = interp1d(pose_t_l, pose_x_l)
    fy     = interp1d(pose_t_l, pose_y_l)
    ftheta = interp1d(pose_t_l, pose_theta_l)

    # align timestamps between poses and img list
    t_count=0
    while img_t_l[t_count]<pose_t_l[0]:
        t_count+=1

    out_pose_l = []
    out_img_id_l = []
    for i in range(t_count, len(img_t_l), iter_):
        img_id = img_id_l[i]
        seq = int(img_id / 1000)
        if seq < seq_start:
            continue
        if seq > seq_end:
            break

        t = img_t_l[i]
        # check exact timestamp of img is within nice pose timestamps for interpolation
        if t<pose_t_l[0] or t>pose_t_l[-1]:
            continue
    
        # interpolate camera pose from the nice poses
        x, y, theta = fx(t), fy(t), ftheta(t)
        rot = theta

        # transformation from boat to world frame
        R_w_b = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0,             0,          1]])
        t_w_b = np.array([x,y,0]) # t: world -> camera

        T_w_b = np.eye(4)
        T_w_b[:3,:3] = R_w_b
        T_w_b[:3,3] =  t_w_b

        T_b_w = np.linalg.inv(T_w_b)
        R_b_w = T_b_w[:3,:3]
        t_b_w = T_b_w[:3,3]
        
   
        # transformation from boat to camera
        R_c_b = np.array([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]])
        t_c_b = 0
        T_c_b = np.eye(4)
        T_c_b[:3,:3] = R_c_b
        T_c_b[:3,3] = t_c_b

        T_b_c = np.linalg.inv(T_c_b)

        T_c_w = np.dot(T_c_b, T_b_w)
        R_c_w = T_c_w[:3,:3]
        t_c_w = T_c_w[:3,3]

        T_w_c = np.linalg.inv(T_c_w)
        
        qw, qx, qy, qz = Quaternion(matrix=R_b_w) # OK
        
        out_pose_l.append([img_id, T_w_c])
    return out_pose_l



def segment_across_season(survey0_id, survey1_id, seq_start, seq_end, iter_):
    
    
    out_root_dir = 'res/%d_%d'%(survey0_id, survey1_id)

    poses0 = np.loadtxt('%s/datasets/poses/%d.txt'%(cst.SCRIPT_DIR, survey0_id))
    poses1 = np.loadtxt('%s/datasets/poses/%d.txt'%(cst.SCRIPT_DIR, survey1_id))


    # sample images of survey0 to segment
    img0_id_v = poses0[:,0]
    img0_seq_v = (img0_id_v/1000).astype(np.int)

    id_start = np.where(img0_seq_v - seq_start > 0)[0][0]
    id_end = np.where(img0_seq_v - seq_end > 0)[0][0]
    #print(poses0[id_start,0])
    #print(poses0[id_end,0])
    #print(np.where(img0_seq_v - seq_start > 0)[0][0])
    #print(np.where(img0_seq_v - seq_end > 0)[0][0])
    
    idx0_v =  np.arange(id_start, id_end, iter_)
    img0_id_v = poses0[idx0_v, 0].astype(np.int)


    # samples images of survey1 NN to images sampled above
    t0 = poses0[idx0_v, 5:]
    t1 = poses1[:, 5:]
    q0 = poses0[idx0_v, 1:5]
    q1 = poses1[:, 1:5]

    t0 = np.expand_dims(t0, 1)
    t1 = np.expand_dims(t1, 0)
    d_t = np.linalg.norm(t0 - t1, ord=None, axis=2)

    match01 = np.argmin(d_t, axis=1)

    
    img0_dir = '%s/%d/'%(cst.SURVEY_DIR, survey0_id)
    img1_dir = '%s/%d/'%(cst.SURVEY_DIR, survey1_id)

    mask0_dir = '%s/%d/water/auto/'%(cst.SEG_DIR, survey0_id)
    mask1_dir = '%s/%d/water/auto/'%(cst.SEG_DIR, survey1_id)

    ## display matches
    #for i, img0_id in enumerate(img0_id_v):
    #    img1_id = poses1[match01[i],0]
    #    img0_fn = '%s/%04d/%04d.jpg'%(img0_dir, img0_id/1000, img0_id%1000)
    #    img1_fn = '%s/%04d/%04d.jpg'%(img1_dir, img1_id/1000, img1_id%1000)
    #    print('%s\n%s\n'%(img0_fn, img1_fn))

    #    img0, img1 = cv2.imread(img0_fn), cv2.imread(img1_fn)
    #    cv2.imshow('img0', img0)
    #    cv2.imshow('img1', img1)
    #    cv2.waitKey(0)


    # segment
    filenames_ims, filenames_segs = [],[]
    filenames_mask = []
    for i, img0_id in enumerate(img0_id_v):
        
        seq = int(img0_id)/1000
        out_dir = '%s/%04d/'%(out_root_dir, seq)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img1_id = poses1[match01[i],0]
        img0_fn = '%s/%04d/%04d.jpg'%(img0_dir, img0_id/1000, img0_id%1000)
        img1_fn = '%s/%04d/%04d.jpg'%(img1_dir, img1_id/1000, img1_id%1000)

        out0_fn = '%s/%d_%d_%04d.png'%(out_dir, i, survey0_id, img0_id)
        out1_fn = '%s/%d_%d_%04d.png'%(out_dir, i, survey1_id, img1_id)

            
        mask0_fn = '%s/%04d/%04d.jpg'%(mask0_dir, seq, img0_id)
        mask1_fn = '%s/%04d/%04d.jpg'%(mask0_dir, seq, img1_id)

        filenames_ims.append(img0_fn)
        filenames_segs.append(out0_fn)
        filenames_mask.append(mask0_fn)

        filenames_ims.append(img1_fn)
        filenames_segs.append(out1_fn)
        filenames_mask.append(mask1_fn)

        print('%s\n%s\n%s\n%s'%(img0_fn, img1_fn, out0_fn, out1_fn))

    
    run_net(filenames_ims, filenames_segs, filenames_mask)



def segment_val(data_id):
    
    filenames_ims, filenames_segs = [],[]

    data_dir = '%s/%d/val/'%(cst.NETVLAD_DATA_DIR, data_id)
    out_root_dir = 'res/%d'%(data_id)

    qImage = np.loadtxt('%s/qImage.txt'%data_dir, dtype=str)
    dbImage = np.loadtxt('%s/dbImage.txt'%data_dir, dtype=str)

    print(np.hstack((qImage, dbImage)).shape)
    for img_fn in np.hstack((qImage, dbImage)):
        out_fn = img_fn.replace('VBags', 'res/%d'%data_id)
        out_dir = os.path.dirname(out_fn)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filenames_ims.append('%s/%s'%(cst.ROOT_DATA_DIR, img_fn))
        filenames_segs.append(out_fn.replace('jpg', 'png'))

        #print('%s/%s'%(cst.ROOT_DATA_DIR, img_fn))
        #print(out_fn)
        #input('\n')
    
    run_net(filenames_ims, filenames_segs, None)


def segment_retrieval_dataset():
    """ """
    res_dir = 'res/icra_retrieval/'
    meta_dir = '%s/lake/meta/icra_retrieval/'%cst.SCRIPT_DIR
    surveys = np.loadtxt('%s/retrieval_surveys.txt'%meta_dir, dtype=np.int32)
    img_ids = np.loadtxt('%s/retrieval_img_ids.txt'%meta_dir, dtype=np.int32)
    surveys = surveys.flatten()
    img_ids = img_ids.flatten()

    filenames_ims, filenames_segs = [],[]
    fn = ['%d/%04d/%04d.jpg'%(s, img_id/1000, img_id%1000) for s,img_id in
            zip(surveys, img_ids)]
    filenames_ims = ['%s/%s'%(cst.SURVEY_DIR, f) for f in fn]
    filenames_segs = ['%s/%s.png'%(res_dir, f.split(".")[0]) for f in fn]

    #print(filenames_ims[:3])
    #print(filenames_segs[:3])

    subdirs = set([os.path.dirname(f) for f in filenames_segs])
    for dir_ in subdirs:
        #print(dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    run_net(filenames_ims, filenames_segs, None)


def segment_retrieval_dataset_with_mask():
    """Img with water filtered out."""
    res_dir = 'res/icra_retrieval/'
    meta_dir= '%s/lake/meta/icra_retrieval/'%cst.SCRIPT_DIR
    mask_dir= '%s/datasets/icra_retrieval/water/global/'%(cst.SCRIPT_DIR)
    img_dir = '%s/datasets/icra_retrieval/img/'%(cst.SCRIPT_DIR)

    surveys = np.loadtxt('%s/retrieval_surveys.txt'%meta_dir, dtype=np.int32)
    img_ids = np.loadtxt('%s/retrieval_img_ids.txt'%meta_dir, dtype=np.int32)
    surveys = surveys.flatten()
    img_ids = img_ids.flatten()

    filenames_ims, filenames_segs, filenames_mask = [],[],[]
    fn = ['%d/%04d/%04d.jpg'%(s, img_id/1000, img_id%1000) for s,img_id in
            zip(surveys, img_ids)]
    filenames_ims  = ['%s/%s'%(img_dir, f) for f in fn]
    #filenames_ims  = ['%s/%s'%(cst.SURVEY_DIR, f) for f in fn]
    filenames_segs = ['%s/%s.png'%(res_dir, f.split(".")[0]) for f in fn]
    filenames_mask = ['%s/%s.png'%(mask_dir, f.split(".")[0]) for f in fn]

    print(filenames_ims[:3])
    print(filenames_segs[:3])
    print(filenames_mask[:3])

    subdirs = set([os.path.dirname(f) for f in filenames_segs])
    for dir_ in subdirs:
        #print(dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_net(filenames_ims, filenames_segs, filenames_mask)


if __name__ == '__main__':
    survey_id = 150429
    seq_start = 2
    seq_end = 32

    #survey_id = 150216
    #seq_start = 1
    #seq_end = 40

    #iter_ = 20
    #segment(survey_id, seq_start, seq_end, iter_)

    #segment_val(4)
    #segment_retrieval_dataset()
    segment_retrieval_dataset_with_mask()


    #survey0_id = 150429
    #survey1_id = 150216
    #seq_start, seq_end = 2, 32
    #segment_across_season(survey0_id, survey1_id, seq_start, seq_end, iter_)


