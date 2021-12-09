#-*- coding: utf-8 -*-
import os
import copy
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from os.path import join as opj
from ensemble_boxes import *

def solve_bbox_problems(bbox_v, scores_v, labels_v):
    """ 
    Solves problems in the "ensemble-boxes" way 
    """
    
    to_remove = np.zeros(bbox_v.shape[0], dtype=np.bool)
    for i in range(bbox_v.shape[0]):
        x1, y1, x2, y2 = bbox_v[i]
        
        if x2 < x1:
#             warnings.warn('X2 < X1 value in box. Swap them.')
            x1, x2 = x2, x1
        if y2 < y1:
#             warnings.warn('Y2 < Y1 value in box. Swap them.')
            y1, y2 = y2, y1
        if x1 < 0:
#             warnings.warn('X1 < 0 in box. Set it to 0.')
            x1 = 0
        if x1 > 1:
#             warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            x1 = 1
        if x2 < 0:
#             warnings.warn('X2 < 0 in box. Set it to 0.')
            x2 = 0
        if x2 > 1:
#             warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            x2 = 1
        if y1 < 0:
#             warnings.warn('Y1 < 0 in box. Set it to 0.')
            y1 = 0
        if y1 > 1:
#             warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            y1 = 1
        if y2 < 0:
#             warnings.warn('Y2 < 0 in box. Set it to 0.')
            y2 = 0
        if y2 > 1:
#             warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
            y2 = 1
        if (x2 - x1) * (y2 - y1) == 0.0:
#             warnings.warn("Zero area box skipped: {}.".format(box_part))
            to_remove[i] = True
    
        bbox_v[i] = x1, y1, x2, y2
    
    if to_remove.sum() > 0:
        # Hack to remove bboxes using min confidence th
        bbox_v[to_remove] = np.array([0.0, 0.0, 1.0, 1.0])
        scores_v[to_remove] = 0.0
        
    return bbox_v, scores_v, labels_v


if __name__ == '__main__':
    # Reading all the predictions (19개)
    subs = [
        # Yolov5 7 models (TTA : Test Time Augmentation)
        pd.read_csv('./results/exp1_tta/wbf_df.csv'),
        pd.read_csv('./results/exp2_tta/wbf_df.csv'),
        pd.read_csv('./results/exp3_tta/wbf_df.csv'),
        pd.read_csv('./results/exp4_tta/wbf_df.csv'),
        pd.read_csv('./results/exp5_tta/wbf_df.csv'),
        pd.read_csv('./results/exp6_tta/wbf_df.csv'),
        pd.read_csv('./results/exp7_tta/wbf_df.csv'),

        # Yolov5 7 models (single image inference)
        pd.read_csv('./results/exp1_single/wbf_df.csv'),
        pd.read_csv('./results/exp2_single/wbf_df.csv'),
        pd.read_csv('./results/exp3_single/wbf_df.csv'),
        pd.read_csv('./results/exp4_single/wbf_df.csv'),
        pd.read_csv('./results/exp5_single/wbf_df.csv'),
        pd.read_csv('./results/exp6_single/wbf_df.csv'),
        pd.read_csv('./results/exp7_single/wbf_df.csv'),

        # CenterNet 4 models (single + TTA)
        pd.read_csv('./results/exp8_centernet_resnet18_dcnv2_140e_coco/wbf_df.csv'),
        pd.read_csv('./results/exp8_centernet_resnet18_dcnv2_140e_coco_tta/wbf_df.csv'),
        pd.read_csv('./results/exp9_centernet_resnet18_dcnv2_140e_coco/wbf_df.csv'),
        pd.read_csv('./results/exp9_centernet_resnet18_dcnv2_140e_coco_tta/wbf_df.csv'),
        
        # Retina 1 models (single)
        pd.read_csv('./results/exp10_retinanet_r101_fpn_1x_coco/wbf_df.csv'),
    ]
    print([len(sub) for sub in subs])
    
    # for i in range(len(subs)):
    #     subs[i] = subs[i].sort_values('score', ascending=False)[:300000].sort_values('file_name')
    
    # Reading original image shapes
    sample_sub = pd.read_csv('data/sample_submission.csv')
    height_list = []
    width_list = []

    for i, name in tqdm(enumerate(sample_sub.file_name)):
        img = opj('data/test_imgs/', Path(name).stem + '.png')
        ori_h, ori_w, _ = cv2.imread(img).shape
        height_list.append(ori_h)
        width_list.append(ori_w)

    
    sample_sub['height'] = height_list
    sample_sub['width'] = width_list
    sample_sub = sample_sub[['file_name', 'height', 'width']]
    sample_sub['file_name'] = sample_sub.file_name.apply(lambda x:Path(x).stem)
        
    height_dict = sample_sub.to_dict('records')
    fnl_dict ={}
    for ix,i in enumerate(height_dict):
        fnl_dict[i['file_name']] = [i['width'],i['height'],i['width'],i['height']]

    # Convert ensemble format
    boxes_dict = {}
    scores_dict = {}
    labels_dict = {}
    whwh_dict = {}

    for idx, i in enumerate(tqdm(sample_sub.file_name.unique())):

        if not i in boxes_dict.keys():
            boxes_dict[i] = []
            scores_dict[i] = []
            labels_dict[i] = []
            whwh_dict[i] = []

        size_ratio = fnl_dict.get(i)
        whwh_dict[i].append(size_ratio)
        tmp_df = [print('Problem') if len(subs[x])==0 else subs[x][subs[x]['file_name']==i] for x in range(len(subs))]
        
        for x in range(len(tmp_df)):
        
            bbox_v = ((tmp_df[x][['x_min','y_min','x_max','y_max']].values) / size_ratio)  # Scaling 0~1 range
            scores_v = tmp_df[x]['score'].values
            labels_v = tmp_df[x]['class_id'].values
            
            bbox_v, scores_v, labels_v = solve_bbox_problems(bbox_v, scores_v, labels_v)   # BBox processing
            
            boxes_dict[i].append(bbox_v.tolist())
            scores_dict[i].append(scores_v.tolist())
            labels_dict[i].append(labels_v.tolist())

    
    # Weighted_boxes_fusion Ensemble (https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
    weights  = [2]*7 + [1]*12  # [1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1]
    iou_thr = 0.31
    skip_box_thr = 0.30
    sigma = 0.1

    fnl = {}
    for i in tqdm(boxes_dict.keys()):
        boxes, scores, labels = weighted_boxes_fusion(boxes_dict[i], scores_dict[i], labels_dict[i],\
                                                        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        if not i in fnl.keys():
            fnl[i] = {'boxes':[],'scores':[],'labels':[]}
            
        fnl[i]['boxes'] = boxes*whwh_dict[i]
        fnl[i]['scores'] = scores
        fnl[i]['labels'] = labels

    # Convert Submission Format
    pd_form = []
    for i in fnl.keys():
        b = fnl[i]
        for j in range(len(b['boxes'])):
            pd_form.append([i,int(b['labels'][j]),round(b['scores'][j],2),
                            int(b['boxes'][j][0]),int(b['boxes'][j][1]),  # xmin, ymin
                            int(b['boxes'][j][2]),int(b['boxes'][j][1]),  # xmax, ymin
                            int(b['boxes'][j][2]),int(b['boxes'][j][3]), # xmax, ymax
                            int(b['boxes'][j][0]),int(b['boxes'][j][3]),]) # xmin, ymax
            

    final_df = pd.DataFrame(pd_form,columns = ['file_name','class_id','confidence',
                                    'point1_x', 'point1_y', 'point2_x', 'point2_y',
                                    'point3_x', 'point3_y', 'point4_x', 'point4_y'])

    print('Total Length', final_df.shape)
    final_df = final_df.sort_values('confidence', ascending=False)[:30000].sort_values('file_name')
    print('Final sub length', final_df.shape)
    final_df['file_name'] = final_df['file_name'].apply(lambda x:x+'.json')
    final_df.to_csv('Final_submission.csv', index=False)
