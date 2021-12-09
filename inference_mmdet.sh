
# Single image inference 3 models (각각 25epoch, 28epoch, 22epoch에서 Best valid score가 나옴)
python inference_mmdet.py --exp_name exp8_centernet_resnet18_dcnv2_140e_coco --checkpoint epoch_25
python inference_mmdet.py --exp_name exp9_centernet_resnet18_dcnv2_140e_coco --checkpoint epoch_28
python inference_mmdet.py --exp_name exp10_retinanet_r101_fpn_1x_coco --checkpoint epoch_22

# TTA 2 models
python inference_mmdet.py --use_tta _tta --exp_name exp8_centernet_resnet18_dcnv2_140e_coco --checkpoint epoch_25
python inference_mmdet.py --use_tta _tta --exp_name exp9_centernet_resnet18_dcnv2_140e_coco --checkpoint epoch_28

