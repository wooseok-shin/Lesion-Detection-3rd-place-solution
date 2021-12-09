#-*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path
from natsort import natsorted
import mmcv
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--dir_prefix', default='')
parser.add_argument('--exp_name', default='retinanet_r101_fpn_1x_coco')
parser.add_argument('--checkpoint', default='exp8',help='model_weight')
parser.add_argument('--use_tta', default='', help="if use_tta: '_tta' else '' ")
args = parser.parse_args()

cfg = Config.fromfile(f'./mmdetection/configs/mmdet_configs/{args.exp_name}{args.use_tta}.py')
cfg.data.test.img_prefix = './data/test_imgs/'
cfg.work_dir = f'./results/{args.dir_prefix}{args.exp_name}/'


if __name__ == '__main__':
    checkpoint = os.path.join('trained_weights', f'{args.checkpoint}.pth')
    model = init_detector(cfg, checkpoint, device='cuda')

    final_list = []
    wbf_list = []
    for i, file in enumerate(tqdm(natsorted(glob(cfg.data.test.img_prefix + '*')))):
        file_name = Path(file).stem
        img = mmcv.imread(file)
        ori_h, ori_w, _ = img.shape
        result = inference_detector(model, img)  # (num_classes, num_boxes, bbox+confidence)
        # show_result_pyplot(model, img, result)

        objects = []
        for class_id, pred in enumerate(result, start=1):
            if len(pred) == 0:
                # print(f'No Predict Class:{class_id}')
                continue
            else:
                for box_id, bbox_score in enumerate(pred):
                    bbox = bbox_score[:4]
                    score = bbox_score[4]
                    x_min = float(round(bbox[0],2))
                    y_min = float(round(bbox[1],2))
                    x_max = float(round(bbox[2],2))
                    y_max = float(round(bbox[3],2))

                    if score < 0.01:
                        continue
                    
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
    os.makedirs(f'results/{args.dir_prefix}{args.exp_name}{args.use_tta}', exist_ok=True)
    submission = submission.sort_values('confidence', ascending=False)[:30000].sort_values('file_name')
    submission.to_csv(f'results/{args.dir_prefix}{args.exp_name}{args.use_tta}/submission.csv', index=False)
    print('Final sub length:', len(submission))


    # 이후 WBF Ensemble을 위한 Format
    wbf_df = pd.DataFrame(wbf_list, columns=['file_name', 'class_id', 'score', 
                                    'x_min', 'y_min', 'x_max', 'y_max',
                                    'width', 'height'])

    wbf_df.to_csv(f'results/{args.dir_prefix}{args.exp_name}{args.use_tta}/wbf_df.csv', index=False)