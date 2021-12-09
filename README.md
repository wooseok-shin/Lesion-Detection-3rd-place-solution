# Lesion-Detection-3rd-place-solution (DACON)
This repository is the 3rd place solution for [DACON Lesion Object Detection AI Contest](https://dacon.io/competitions/official/235855/overview/description).

## Requirements (make_dataset.sh에서 관련 라이브러리를 설치함.)
- Ubuntu 18.04, Cuda 11.1
- Anaconda - Python 3.8 (Anaconda 가상환경, Yolov5 및 MMdetection 학습 시 따로 구축하는 것을 권장하나 제 환경에서는 한 번에 돌아가긴 했습니다.)
- numpy
- pandas
- opencv-python
- python-dateutil
- pytz
- six
- matplotlib
- natsort
- tqdm
- scikit-learn
- ensemble-boxes
- torch==1.9.0 torchvision 0.10.0 with cuda 11.1 (YoloV5)
- torch==1.8.0 torchvision 0.9.0 with cuda 11.1 (MMDetection)
- mmcv-full
- mmdet

## Directory 구조
```bash
├── README.md
├── data
│   ├── class_id_info.csv
│   ├── lesion.names
│   ├── sample_submission.csv
│   └── train
        ├── train_100000.json
        ├── train_100001.json
        ├── ...
│   └── test
        ├── test_200000.json
        ├── test_200001.json
        ├── ...
├── trained_weights (학습 완료된 가중치)
│   ├── exp1.pt
│   ├── exp2.pt
│   ├── ...

├── pretrained_weights (centernet, retinanet -> 포함하여 제출함 or MMdetection Github로부터 받을 수 있음.)
│   ├── centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth
│   ├── retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth
│   ├── ...


```

## Yolov5, MMdetection Dataset 구축 (+ install yolov5)
- 주어진 데이터를 Yolov5와 MMdetection 학습에 필요한 각각의 포맷으로 변환시켜줍니다.
```bash
sh make_dataset.sh
```

## Train YoloV5 models (Yolov5 L,X)
- Ensemble 시 다양성을 주기위해 이미지 사이즈, 모델 사이즈, Fold를 다양하게 학습하였습니다.
- 그리고 적절한 구조 및 하이퍼파라미터를 찾는것에 시간을 소모하여 0Fold 모델이 많습니다.
- 따라서, 5Fold CV방식의 학습을 수행하지는 않았습니다.
```bash
sh train_yolov5.sh
```

## Install MMdetection
- 기존에 Yolov5를 학습시킬 때 torch 1.9.0 버전을 사용하였는데, MMDetection에서 1.9.0이 돌아가지 않는 현상이 있었습니다.
- 따라서, torch를 1.8.0, torchvision을 0.10.0으로 재설치 해주었습니다. (가상환경을 따로할 시 상관없음)
```bash
sh mmdet_install.sh
```

## Train MMdetection models (CenterNet, RetinaNet)
- 처음엔 Yolov5 계열로만 Ensemble을 하다가 성능 향상이 더뎌진 것 같아 전혀 다른 구조의 모델을 사용하였습니다.
- 그 중에서 CenterNet(R-18)과 RetinaNet(R-101)을 학습하였습니다.
- Backbone으로 CBNetV2나 Detector로 HTC 등 COCO 기준으로 더 좋은 성능을 보이는 모델도 학습해보았으나 오버피팅 문제가 발생하여 가벼운 구조를 채택하였습니다.
```bash
sh train_mmdet.sh
```


## Inference YoloV5 models
```bash
sh inference_yolo.sh
```

## Inference MMDet models
```bash
sh inference_mmdet.sh
```

## WBF Ensemble - 7 Yolo models x 2(NoTTA, TTA) + 2 CenterNet x 2(NoTTA, TTA) + 1 RetinaNet
- Make final submission
```python
python ensemble_wbf.py
```


## Public, Private Score 복원: 학습된 모델 Weight를 불러와서 inference하기 (Github으로부터 Weights Download)
- 10개의 Weight를 불러와야하는데 Github 서버 문제인지 가끔 하나씩 안불러오는 경우가 있는 것 같습니다.
- exp1~exp10까지 다 불러와졌는지 확인이 필요합니다.
- 한 두개가 빠져있으면 해당 폴더를 삭제하고 다시 실행하거나 wget으로 하나씩 불러오면 됩니다.
```bash
sh inference_load_weight.sh
```

