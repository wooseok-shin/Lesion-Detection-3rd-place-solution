#-*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join as opj

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='')
parser.add_argument('--dir_prefix', default='yolov5/runs/detect/')
parser.add_argument('--test_img_path', default='data/test_imgs/')
parser.add_argument('--threshold', type=float, default=0.001)
args = parser.parse_args()


if __name__ == '__main__':
    final_list = []
    wbf_list = []
    preds_list = glob(f'{args.dir_prefix}/{args.exp_name}/labels/*.txt')

    for pred in tqdm(preds_list):
        file_name = Path(pred).stem
        img_path = opj(args.test_img_path, file_name + '.png')
        ori_h, ori_w, _ = cv2.imread(img_path).shape
        pred = open(pred).readlines()

        objects = []
        for line in pred:
            splitted_line = line.split()   # [cls, x, y, w, h, confidence]
            class_id = int(splitted_line[0]) + 1

            x_center = float(splitted_line[1])
            y_center = float(splitted_line[2])
            width = float(splitted_line[3])
            height = float(splitted_line[4])
            score = float(splitted_line[5])

            if score < args.threshold:
                continue

            int_x_center = int(ori_w * x_center)
            int_y_center = int(ori_h * y_center)
            int_width = int(ori_w * width)
            int_height = int(ori_h * height)

            x_min = int_x_center - int_width / 2
            y_min = int_y_center - int_height / 2
            x_max = x_min + int_width
            y_max = y_min + int_height

            sub_list = [file_name + '.json', class_id, score, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            final_list.append(sub_list)

            # For wbf ensemble
            wbf = [file_name, class_id, score, x_min, y_min, x_max, y_max, ori_w, ori_h]
            wbf_list.append(wbf)

    # Submission Format
    submission = pd.DataFrame(final_list, columns=['file_name', 'class_id', 'confidence', 
                                    'point1_x', 'point1_y', 'point2_x', 'point2_y',
                                    'point3_x', 'point3_y', 'point4_x', 'point4_y'])
    print('Full sub length:', len(submission))
    # 최대 30000줄까지 기록 (confidence score를 기준으로 slicing)
    submission = submission.sort_values('confidence', ascending=False)[:30000].sort_values('file_name')
    submission.to_csv(f'results/{args.exp_name}/submission.csv', index=False)
    print('Final sub length:', len(submission))
    
    # 이후 WBF Ensemble을 위한 Format
    wbf_df = pd.DataFrame(wbf_list, columns=['file_name', 'class_id', 'score', 
                                    'x_min', 'y_min', 'x_max', 'y_max',
                                    'width', 'height'])

    os.makedirs(f'results/{args.exp_name}', exist_ok=True)
    wbf_df.to_csv(f'results/{args.exp_name}/wbf_df.csv', index=False)