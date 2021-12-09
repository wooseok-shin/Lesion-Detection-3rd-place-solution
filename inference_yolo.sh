cd yolov5

# yolov5/results/expN 폴더에 txt파일 예측 결과물 저장

## Single image inference 7 models
python detect.py --weights weights/exp1/weights/epoch120.pt --project ../results/ --name exp1_single --img 384 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave
python detect.py --weights weights/exp2/weights/epoch150.pt --project ../results/ --name exp2_single --img 512 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave
python detect.py --weights weights/exp3/weights/epoch120.pt --project ../results/ --name exp3_single --img 320 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave
python detect.py --weights weights/exp4/weights/epoch120.pt --project ../results/ --name exp4_single --img 384 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave
python detect.py --weights weights/exp5/weights/epoch120.pt --project ../results/ --name exp5_single --img 480 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave
python detect.py --weights weights/exp6/weights/epoch120.pt --project ../results/ --name exp6_single --img 480 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave
python detect.py --weights weights/exp7/weights/epoch120.pt --project ../results/ --name exp7_single --img 640 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave

## TTA 7 models
python detect.py --weights weights/exp1/weights/epoch120.pt --project ../results/ --name exp1_tta --img 384 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment
python detect.py --weights weights/exp2/weights/epoch150.pt --project ../results/ --name exp2_tta --img 512 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment
python detect.py --weights weights/exp3/weights/epoch120.pt --project ../results/ --name exp3_tta --img 320 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment
python detect.py --weights weights/exp4/weights/epoch120.pt --project ../results/ --name exp4_tta --img 384 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment
python detect.py --weights weights/exp5/weights/epoch120.pt --project ../results/ --name exp5_tta --img 480 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment
python detect.py --weights weights/exp6/weights/epoch120.pt --project ../results/ --name exp6_tta --img 480 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment
python detect.py --weights weights/exp7/weights/epoch120.pt --project ../results/ --name exp7_tta --img 640 --source ../data/test_imgs --save-txt --save-conf --conf-thres 0.001 --iou-thres 0.6 --nosave --augment

cd ../

# Yolov5 infernece format to submission format
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