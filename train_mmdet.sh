
# Train three models
cp -r mmdet_configs/ mmdetection/configs/    # 수정해놓은 config 폴더 복사해서 mmdetection/configs 폴더에 넣기

# Download pretrain weight from mmdetection github (CenterNet R-18 and Retina ResNet-101)
wget https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth -P ./pretrain_weights/
wget https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth -P ./pretrain_weights/

cd mmdetection

## 1.Centernet Fold 0
bash tools/dist_train.sh configs/mmdet_configs/exp8_centernet_resnet18_dcnv2_140e_coco.py 1

## 2.Centernet Fold 1
bash tools/dist_train.sh configs/mmdet_configs/exp9_centernet_resnet18_dcnv2_140e_coco.py 1

## 3.Retina ResNet-101 Fold 0
bash tools/dist_train.sh configs/mmdet_configs/exp10_retinanet_r101_fpn_1x_coco.py 1
