
import numpy as np
import cv2

ignore_label = 255


#labels = [
#    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#,
#    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#]
#


label_name = [
    'road'                 ,
    'sidewalk'             ,
    'building'             ,
    'wall'                 ,
    'fence'                ,
    'pole'                 ,
    'traffic light'        ,
    'traffic sign'         ,
    'vegetation'           ,
    'terrain'              ,
    'sky'                  ,
    'person'               ,
    'rider'                ,
    'car'                  ,
    'truck'                ,
    'bus'                  ,
    'train'                ,
    'motorcycle'           ,
    'bicycle'              
]

# rgb
palette = [[128, 64, 128], 
        [244, 35, 232], 
        [70, 70, 70], 
        [102, 102, 156], 
        [190, 153, 153], 
        [153, 153, 153], 
        [250, 170, 30],
        [220, 220, 0], 
        [107, 142, 35], 
        [152, 251, 152], 
        [70, 130, 180], 
        [220, 20, 60], 
        [255, 0, 0], 
        [0, 0, 142], 
        [0, 0, 70],
        [0, 60, 100], 
        [0, 80, 100], 
        [0, 0, 230], 
        [119, 11, 32]]


palette_bgr = [ [l[2], l[1], l[0]] for l in palette]



# my colorspace
from matplotlib import cm
viridis = cm.get_cmap('plasma', 12)
viridis = cm.get_cmap('viridis', 12)
viridis = cm.get_cmap('inferno', 12)
viridis = cm.get_cmap('magma', 12)
viridis = cm.get_cmap('cividis', 12)
#print(viridis)
#print('viridis.colors', viridis.colors)
#print('viridis(range(12))', viridis(range(12)))
#print('viridis(np.linspace(0, 1, 12))', viridis(np.linspace(0, 1, 12)))
num_color = 50
myjet = (255 * viridis(np.linspace(0, 1, num_color))).astype(np.uint8)
myjet = myjet[:,:3]
myjet = myjet[:,::-1]


def gen_jet(color_num):

    b = (255*np.linspace(0, 1, color_num)).astype(np.uint8)
    g = (255*np.linspace(0, 1, color_num)).astype(np.uint8)
    r = (255*np.linspace(0, 1, color_num)).astype(np.uint8)

    r = r[::-1]

    bgr = np.vstack((b,g,r)).T
    return bgr


if __name__=='__main__':
    #print(myjet)
    #np.savetxt('meta/color_map.txt', np.array(palette), fmt='%d')

    bgr = gen_jet(50)
    print(bgr)
    
    
    #h = 40
    #num_class = len(label_name)
    #img_size = h*num_class
    #img = np.ones((img_size,img_size,3))*255
    #color_bar = np.ones((num_class*h,h,3))
    #for i in range(num_class):
    #    begin = i*h
    #    end = begin + h
    #    color_bar[begin:end,:] = [palette[i][2], palette[i][1], palette[i][0]]
    #    text_pos_x = h
    #    text_pos_y = begin+int(0.7*h)
    #    cv2.putText(img, str(i) + '-' + label_name[i], (text_pos_x,text_pos_y), fontFace=0, fontScale=0.5, color=(0,0,0))
    #img[:,:h] = color_bar
    #cv2.imwrite('meta/color_bar.png', img)


    
