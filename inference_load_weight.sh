
# Download trained weight from my github (https://github.com/wooseok-shin/Lesion-Detection-3rd-place-solution)
wget -i https://raw.githubusercontent.com/wooseok-shin/Lesion-Detection-3rd-place-solution/main/load_trained_weight.txt -P trained_weights


# Yolov5 7x2 models
cd yolov5
# yolov5/results/expN 폴더에 txt파일 예측 결과물 저장

## Single image inference 7 models
python detect.py --weights ../trained_weights/exp1.pt --project ../results/ --name exp1_single --img 384 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6
python detect.py --weights ../trained_weights/exp2.pt --project ../results/ --name exp2_single --img 512 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6
python detect.py --weights ../trained_weights/exp3.pt --project ../results/ --name exp3_single --img 320 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6
python detect.py --weights ../trained_weights/exp4.pt --project ../results/ --name exp4_single --img 384 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6
python detect.py --weights ../trained_weights/exp5.pt --project ../results/ --name exp5_single --img 480 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6
python detect.py --weights ../trained_weights/exp6.pt --project ../results/ --name exp6_single --img 480 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6
python detect.py --weights ../trained_weights/exp7.pt --project ../results/ --name exp7_single --img 640 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6

## TTA 7 models
python detect.py --weights ../trained_weights/exp1.pt --project ../results/ --name exp1_tta --img 384 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment
python detect.py --weights ../trained_weights/exp2.pt --project ../results/ --name exp2_tta --img 512 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment
python detect.py --weights ../trained_weights/exp3.pt --project ../results/ --name exp3_tta --img 320 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment
python detect.py --weights ../trained_weights/exp4.pt --project ../results/ --name exp4_tta --img 384 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment
python detect.py --weights ../trained_weights/exp5.pt --project ../results/ --name exp5_tta --img 480 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment
python detect.py --weights ../trained_weights/exp6.pt --project ../results/ --name exp6_tta --img 480 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment
python detect.py --weights ../trained_weights/exp7.pt --project ../results/ --name exp7_tta --img 640 --source ../data/test_imgs --save-txt --save-conf --nosave --conf-thres 0.001 --iou-thres 0.6 --augment

cd ../

## Yolov5 infernece format to submission format
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp1_single --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp2_single --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp3_single --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp4_single --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp5_single --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp6_single --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp7_single --test_img_path data/test_imgs/

python inference_yolo_agg.py --dir_prefix results/ --exp_name exp1_tta --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp2_tta --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp3_tta --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp4_tta --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp5_tta --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp6_tta --test_img_path data/test_imgs/
python inference_yolo_agg.py --dir_prefix results/ --exp_name exp7_tta --test_img_path data/test_imgs/


# MMdetection 5 models
cp -r mmdet_configs/ mmdetection/configs/ # Configs 복사 및 덮어쓰기

## Single image inference 3 models
python inference_load_mmdet.py --exp_name exp8_centernet_resnet18_dcnv2_140e_coco --checkpoint exp8
python inference_load_mmdet.py --exp_name exp9_centernet_resnet18_dcnv2_140e_coco --checkpoint exp9
python inference_load_mmdet.py --exp_name exp10_retinanet_r101_fpn_1x_coco --checkpoint exp10

## TTA 2 models
python inference_load_mmdet.py --use_tta _tta --exp_name exp8_centernet_resnet18_dcnv2_140e_coco --checkpoint exp8
python inference_load_mmdet.py --use_tta _tta --exp_name exp9_centernet_resnet18_dcnv2_140e_coco --checkpoint exp9

# Final WBF Ensemble
python ensemble_wbf.py