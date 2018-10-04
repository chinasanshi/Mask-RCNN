#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    #utils.download_trained_weights(COCO_MODEL_PATH)
    # 直接使用 wget 获取模型，模型大约 246M ，若失败需要重新下载
    os.system("wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5")



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)




# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# 读取图片，并检测和分割；如果需要使用可以放开下面的注释
#image = skimage.io.imread(sys.argv[1]))
#results = model.detect([image], verbose=1)
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                            class_names, r['scores'])


# 捕捉视频帧
cap = cv2.VideoCapture(sys.argv[1])
#保存视频

# 视频的宽度
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
width = 640
# 视频的高度
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
height = 480
# 视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 视频的编码
#fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) # not support YUYV
#fourcc = cv2.VideoWriter_fourcc(*'XVID') # not support XVID, but can work
fourcc = cv2.VideoWriter_fourcc(*"MPEG")# not support MPEG, but can work

# 定义视频输出
videoSave = cv2.VideoWriter("out.mp4", fourcc, fps, (width*2, height))

frame_num = 0
results = 0

while True:
    _, frame = cap.read()

    frame_num += 1

    frame = cv2.resize(frame, (width, height))
    org = frame.copy()

    if frame_num % 5 == 0  :
        # Run detection
        results = model.detect([frame], verbose=1)

    # Visualize results
    r = results[0]
    frame = visualize.ret_display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    frame = np.concatenate((org,frame), axis = 1)

    cv2.imshow("Frame", frame)
    #cv2.imwrite("mask.jpg", frame)
    videoSave.write(frame)

    #if cv2.waitKey(33):
    #    break


cv2.destroyAllWindows()




