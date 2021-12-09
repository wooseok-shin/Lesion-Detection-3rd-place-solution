#-*- coding: utf-8 -*-
import os
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold

def convert_to_coco(json_paths):
    """
        only for train dataset
    """
    res = defaultdict(list)
    
    categories = {
        '01_ulcer': 1,
        '02_mass': 2,
        '04_lymph': 3,
        '05_bleeding': 4
    }
    
    n_id = 0
    for json_path in tqdm(json_paths):
        with open(json_path, 'r') as f:
            tmp = json.load(f)
            
        image_id = int(tmp['file_name'].split('_')[-1][:6])
        res['images'].append({
            'id': image_id,
            'width': tmp['imageWidth'],
            'height': tmp['imageHeight'],
            'file_name': Path(tmp['file_name']).stem + '.png',
        })
        
        for shape in tmp['shapes']:
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][2]
            
            w, h = x2 - x1, y2 - y1
            
            res['annotations'].append({
                'id': n_id,
                'image_id': image_id,
                'category_id': categories[shape['label']],
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0,
            })
            n_id += 1
    
    for name, id in categories.items():
        res['categories'].append({
            'id': id,
            'name': name,
        })
    return res

if __name__ == '__main__':
    # Full json (no split json for yolov5)
    os.makedirs('data/annos/', exist_ok=True)
    json_paths = np.array(glob(os.path.join('data', 'train', '*.json')))
    train_json = convert_to_coco(json_paths)
    with open(f'./data/annos/train_annotations_full.json', 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=True, indent=4)

    # Train/Valid 10Fold Split (Make total json file)
    n_splits=10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    json_paths = np.array(glob(os.path.join('data', 'train', '*.json')))

    for fold, (trn_idx, val_idx) in enumerate(kf.split(json_paths)):
        val_json = json_paths[val_idx]
        train_json = json_paths[trn_idx]
        # print(len(train_json), len(val_json))

        train_json = convert_to_coco(train_json)
        val_json = convert_to_coco(val_json)

        with open(f'./data/annos/train_annotations_{n_splits}split_{fold}fold.json', 'w', encoding='utf-8') as f:
            json.dump(train_json, f, ensure_ascii=True, indent=4)
        with open(f'./data/annos/valid_annotations_{n_splits}split_{fold}fold.json', 'w', encoding='utf-8') as f:
            json.dump(val_json, f, ensure_ascii=True, indent=4)
        
        # MMdetection 모델은 0,1 Fold만 학습에 사용함.
        if fold == 1:
            break
            
    # Make Train PNG images
    json_paths = np.array(glob(os.path.join('data', 'train', '*.json')))
    save_path = './data/train_imgs'
    os.makedirs(save_path, exist_ok=True)
    for json_file in tqdm(json_paths):
        with open(json_file, 'r') as f:
            name = Path(json_file).stem
            json_file = json.load(f)
            image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(save_path, name + '.png'), image)

    # Make Test PNG images
    json_paths = np.array(glob(os.path.join('data', 'test', '*.json')))
    save_path = './data/test_imgs'
    os.makedirs(save_path, exist_ok=True)
    for json_file in tqdm(json_paths):
        with open(json_file, 'r') as f:
            name = Path(json_file).stem
            json_file = json.load(f)
            image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(save_path, name + '.png'), image)