# Train seven models

## Ensemble 시 다양성을 많이 주기위해 이미지 사이즈, 모델 사이즈, Fold를 다양하게 학습
cd yolov5
mkdir weights/

### 2 x yolov5l (0Fold: 384,512 size)
python train.py --project weights/ --name=exp1 --img 384 --batch 16 --epochs 125 --data lesion_0f.yaml --weights yolov5l.pt --save-period 5 --workers 4
python train.py --project weights/ --name=exp2 --img 512 --batch 16 --epochs 200 --data lesion_0f.yaml --weights yolov5l.pt --save-period 5 --workers 4

### 5 x yolov5x (0Fold: 320,384,480 size & 2,4Fold)
python train.py --project weights/ --name=exp3 --img 320 --batch 16 --epochs 125 --data lesion_0f.yaml --weights yolov5x.pt --save-period 5 --workers 4
python train.py --project weights/ --name=exp4 --img 384 --batch 16 --epochs 125 --data lesion_0f.yaml --weights yolov5x.pt --save-period 5 --workers 4
python train.py --project weights/ --name=exp5 --img 480 --batch 16 --epochs 125 --data lesion_0f.yaml --weights yolov5x.pt --save-period 5 --workers 4
python train.py --project weights/ --name=exp6 --img 480 --batch 16 --epochs 125 --data lesion_2f.yaml --weights yolov5x.pt --save-period 5 --workers 4
python train.py --project weights/ --name=exp7 --img 640 --batch 16 --epochs 125 --data lesion_4f.yaml --weights yolov5x.pt --save-period 5 --workers 4

