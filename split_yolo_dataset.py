#-*- coding: utf-8 -*-
import os
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from sklearn.model_selection import KFold


def copy_files(files, prefix):
    os.makedirs(prefix, exist_ok=True)
    for instance in tqdm(files):
        file_name = os.path.basename(instance)
        save_path = os.path.join(prefix, file_name)
        shutil.copy(instance, save_path)


if __name__ == '__main__':
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    full_images = np.array(natsorted(glob('./data/train_imgs/*.png')))
    full_labels = np.array(natsorted(glob('./data/yolo_labels/*.txt')))
    
    use_folds = [0,2,4]   # 10Fold중 0,2,4 Fold만 사용함. (학습 시간 때문에)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(full_images)):
        if fold not in use_folds:
            continue
        
        val_images = full_images[val_idx]
        train_images = full_images[trn_idx]

        val_labels = full_labels[val_idx]
        train_labels = full_labels[trn_idx]
        
        copy_files(train_images, prefix=f'./data/yolo/{fold}fold/train/images/')
        copy_files(train_labels, prefix=f'./data/yolo/{fold}fold/train/labels/')
        copy_files(val_images, prefix=f'./data/yolo/{fold}fold/valid/images/')
        copy_files(val_labels, prefix=f'./data/yolo/{fold}fold/valid/labels/')