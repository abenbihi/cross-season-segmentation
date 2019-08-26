import numpy as np

d = np.array([-0.295703, 0.157403, -0.001469, -0.000924])
K = np.array([
    [780.170806, 0, 317.745657],
    [0, 708.378535, 246.801583],
    [0, 0, 1]])
K_inv =np.linalg.inv(K)

MIN_MATCH_COUNT = 10


# lake img
MACHINE = 1
if MACHINE == 0:
    ROOT_DATA_DIR = '/mnt/data_drive/dataset/lake/'
    POSE_DIR = '/home/abenbihi/ws/datasets/lake/poses/'
    WS_DIR = '/home/abenbihi/ws/'

elif MACHINE == 1:
    ROOT_DATA_DIR = '/mnt/lake/'
    WS_DIR = '/home/gpu_user/assia/ws/'

else:
    print("Get you MTF MACHINE macro correct")
    exit(1)

SURVEY_DIR = '%s/VBags/'%ROOT_DATA_DIR
PAIRS_DIR = '%s/Pairs/'%ROOT_DATA_DIR
POSE_DIR = '%s/MapsDyn'%ROOT_DATA_DIR
SCRIPT_DIR = '%s/datasets/lake'%WS_DIR
SEG_DIR = '%s/datasets/seg'%SCRIPT_DIR

LABEL = {'mask':0, 'water':1, 'sky':2, 'vegetation':3}
COLOR = {'mask':[0,0,0], 'water':[255,0,0], 'sky':[255,255,0],
        'vegetation':[0,255,0]}

W = 700


# CMU-Seasons
# colmap output dir
MACHINE = 0
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
NUM_CLASS = 19

