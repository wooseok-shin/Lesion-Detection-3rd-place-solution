pip install Pillow cycler kiwisolver numpy pandas opencv-python python-dateutil pytz six matplotlib natsort tqdm scikit-learn ensemble-boxes
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 1. Make coco format (for mmdetection & yolov5)
python convert_coco_format.py

# 2. Coco to Yolov5 format dataset
git clone https://github.com/ssaru/convert2Yolo.git
cd convert2Yolo
mkdir ../data/yolo_labels/
## lesion.names 파일을 ../data/ 위치에 만들어주어야 함. (현재는 만들어져있음.)
python example.py --datasets COCO --img_path ../data/train_imgs/ --label ../data/annos/train_annotations_full.json --convert_output_path ../data/yolo_labels/ --img_type ".png" --manifest_path ./ --cls_list_file ../data/lesion.names
cd ..

# 3. Yolov5 train/valid split
python split_yolo_dataset.py

# 4. Yolov5 clone & Prerequisite
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cp ../etc/* ./data/    # Yolov5 학습을 위해 yolov5/data/경로에 yaml 파일을 만들어 주어야함 (학습에 필요한 경로 설정) - 미리 첨부해놓은 파일 복사해서 해당 폴더에 넣어주기
