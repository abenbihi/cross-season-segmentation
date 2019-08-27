
import os
import cv2
import numpy as np

import palette
import cst

def col2lab(col):
    """
    Convert color map to label map
    """

    lab = np.zeros(col.shape[:2]).astype(np.uint8)
    for i, color in enumerate(palette.palette):
        #print('%d: %s'%(i, palette.label_name[i]), color)
        color = [color[2], color[1], color[0]]
        
        # I know, this is ugly 
        mask = np.zeros(col.shape[:2]).astype(np.uint8)
        mask = 255*(col==color).astype(np.uint8)
        mask = (np.sum(mask,axis=2) == (255*3)).astype(np.uint8)
        #l, c = np.where(mask==1)
        lab[mask==1] = i
        #lab[l,c] = i

        #cv2.imshow('mask', mask)
        #stop_show = cv2.waitKey(0) & 0xFF
        #if stop_show == ord("q"):
        #    exit(0)

    return lab


def lab2col(lab, colors):
    """
    Convert label map to color map
    """
    col = np.zeros((lab.shape + (3,))).astype(np.uint8)
    labels = np.unique(lab)

    if np.max(labels) >= len(colors):
        print("Error: you need more colors np.max(labels) >= len(colors): %d >= %d"
                %(np.max(labels), len(colors)) )
        exit(1)

    for i, label in enumerate(labels):
        color = colors[i] 
        col[lab==i,:] = color
    return col


def extract_connected_components(lab0):
    """
    Maybe use
    https://stackoverflow.com/questions/26332883/how-to-find-all-connected-components-in-a-binary-image-in-matlab
    """

    color_palette = palette.gen_jet(50)
    
    cc = np.zeros(lab0.shape).astype(np.uint8)
    next_cc_id = 1

    lab = lab0.copy()
    lab_id_l = np.unique(lab)
    print('Label presents: ', lab_id_l)

    for lab_id in lab_id_l:
    #for lab_id in range(10,11):

        # mask all pixels with label lab_id
        l,c = np.where(lab==lab_id)
        #print(l)
        #print(c)

        while len(l) != 0:
            mask = np.zeros(lab.shape).astype(np.uint8)
            mask[l,c] = 255
            cv2.imshow('mask', mask)
        
            # output of flood fill, 1 where it has filled
            # I don't know why, it does not want to fill with 128
            mask_out = np.zeros((lab.shape[0]+2, lab.shape[1]+2)).astype(np.uint8)
            a = cv2.floodFill(mask, mask_out, (c[0], l[0]), 128)

            # save new connected component
            cc[mask_out[1:-1, 1:-1]==1] = next_cc_id
            next_cc_id += 1
            
            # clear pixels from this connected component
            lab[mask_out[1:-1, 1:-1]==1] = -1
            l, c = np.where(lab==lab_id)

            ## color display of flood fill
            #if (1==1):
            #    print('cc_id: %d'%(next_cc_id-1))
            #    mask_out_col = np.zeros((mask_out.shape) + (3,)).astype(np.uint8)
            #    mask_out_col[mask_out==1,:] = color_palette[lab_id]
            #    cv2.imshow('mask_out', mask_out)
            #    cv2.imshow('mask_out_col', mask_out_col)
            #    stop_show = cv2.waitKey(0) & 0xFF
            #    if stop_show == ord("q"):
            #        exit(0)


    # color display of all connected components
    cc_col = lab2col(cc, color_palette)
    #cc_col[cc==0] = 0
    cv2.imshow('cc_col', cc_col)
    stop_show = cv2.waitKey(0) & 0xFF
    if stop_show == ord("q"):
        exit(0)





def sem2lab(slice_id, cam_id, survey_id, mode):
    """
    Convert semantic seg dir to lab 
    """

    if mode == 'database':
        res_dir = 'res/ext_cmu/%d/c%d_db/' %(slice_id, cam_id)
        meta_fn = '%s/surveys/%d/fn/c%d_db.txt' %(cst.META_DIR, slice_id, cam_id)
    elif mode == 'query':
        res_dir = 'res/%d/%d_%d/' %(slice_id, cam_id, survey_id)
        meta_fn = '%s/surveys/%d/fn/c%d_%d.txt' %(cst.META_DIR, slice_id, cam_id, survey_id)

    if not os.path.exists(res_dir):
        print("Error: you did not segment the images for slice | cam | survey_id: %d | %d | %d"
                %(slice_id, cam_id, survey_id))
        exit(1)
    
    # get all file names
    meta = [ll.split("\n")[0] for ll in open(meta_fn, 'r').readlines()]
    img_fn_l = ['%s/slice%d/%s/%s'%(cst.EXT_IMG_DIR, slice_id, mode, l) for l in meta]
    col_fn_l = ['%s/col/%s.png'%(res_dir, l) for l in meta]
    
    for img_fn, col_fn in zip(img_fn_l, col_fn_l):
        print(img_fn)
        img = cv2.imread(img_fn)[:,:cst.W]
        col = cv2.imread(col_fn)
        cv2.imshow(' img | col', np.hstack((img, col)))

        lab = col2lab(col)
        #cv2.imshow('lab', lab)
        #col2 = lab2col(lab)
        #cv2.imshow('col2', col2)

        extract_connected_components(lab)

        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)



def sem2graph():
    """
    Semantic segmentation to graph.
    """


if __name__=='__main__':
    #sem2graph()
    slice_id    = 24
    cam_id      = 0
    survey_id   = 'db'
    sem2lab(slice_id, cam_id, -1, 'database')
