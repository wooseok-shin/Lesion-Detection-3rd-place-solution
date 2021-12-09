# Install mmcv and mmdet
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmdet

# MMDetection을 처음부터 다운로드 받아서 할 시 Git Clone 수행
git clone https://github.com/open-mmlab/mmdetection.git


## or
# cd mmdetection
# pip install -r requirements/build.txt
# pip install -v -e .  # or "python setup.py develop"

