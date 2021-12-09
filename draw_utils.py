#-*- coding: utf-8 -*-
import os
import cv2
import json
import shutil
import base64
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict

def get_colors(classes: List) -> Dict[str, tuple]:
    return {c: tuple(map(int, np.random.randint(0, 255, 3))) for c in classes}

def draw_bbox(
    json_path: os.PathLike,
    coco_path: os.PathLike, 
    save_path: os.PathLike,
    n_images: int = 10,
) -> None:
    with open(coco_path, 'r') as f:
        ann_json = json.load(f)
        
    images = [{v['id']: v['file_name']} for v in ann_json['images']]
    categories = {v['id']: v['name'] for v in ann_json['categories']}
    
    ann = defaultdict(list)
    for a in ann_json['annotations']:
        bbox = list(map(round, a['bbox']))
        ann[a['image_id']].append(
            {
                'category_id': categories.get(a['category_id']),
                'bbox': bbox,
            }
        )
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        
    colors = get_colors(categories.values())
    for v in tqdm(images[:n_images]):
        image_id, file_name = list(v.items())[0]
        file_path = os.path.join(json_path, file_name)
        with open(file_path, 'r') as f:
            json_file = json.load(f)
            
        image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        annots = ann[image_id]
        
        for a in annots:
            label = a['category_id']
            x1, y1, w, h = a['bbox']
            x2, y2 = x1 + w, y1 + h
            
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1-20), (x1+tw, y1), colors[label], -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        file_name = file_name.split('.')[0] + '.jpg'
        cv2.imwrite(os.path.join(save_path, file_name), image)