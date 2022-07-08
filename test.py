import os

import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
import json
import numpy as np
import heapq
import colorsys
import re
import random
from typing import List, Optional
import math
import click
import PIL.Image
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm


coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

cocoRoot = '/home/dmsheng/demo/try/mmdetection/data/coco_5_aug_refine'
dataType = "train2017" # ["train2017", "val2017", "test2017"]
data_dir = os.path.join(cocoRoot, dataType)
annFile = os.path.join(cocoRoot, f'annotations/instances_{dataType}.json')

def test():
    # Specify the path to model config and checkpoint file
    config_file = '/home/dmsheng/demo/try/mmdetection/work_dirs/vfnet_x101_64x4d_fpn_mstrain_2x_coco_center_5_aug/vfnet_x101_64x4d_fpn_mstrain_2x_coco_center_5_aug.py'
    checkpoint_file = '/home/dmsheng/demo/try/mmdetection/work_dirs/vfnet_x101_64x4d_fpn_mstrain_2x_coco_center_5_aug/latest.pth'
    save_dir = '/home/dmsheng/demo/try/mmdetection/experiments/vfnet/crop_aug/'

    os.makedirs(save_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:5')

    # test a single image and show the results
    imgs = sorted(glob('data/coco_5_default/val2017/' + '*'))

    for img in tqdm(iter(imgs)):
        result = inference_detector(model, img)
        name = img.split('/')[-1]

        # or save the visualization results to image files
        model.show_result(img, result, out_file=os.path.join(save_dir, name))

def scores():
    result = '/home/dmsheng/demo/try/mmdetection/deformable_detr_coco_test.bbox.json'

    val=json.load(open(result, 'r'))
    scores = np.zeros(91)
    id = np.zeros(len(val))
    for i in range(len(val)):
        id[i] = val[i]['image_id']
        scores[int(id[i])] += val[i]['score']

    js = []
    for i in range(len(scores)-1):
        js_i = {'id': f'{i+1}', 'score': f'{scores[i+1]}'}
        js.append(js_i)
    jsondata = json.dumps(js,separators=(',', ':'))
    f = open('deformable_detr_coco_test.json', 'w')
    f.write(jsondata)
    f.close()

def plot():
    result = '/home/dmsheng/demo/try/yolox_coco_test.json'
    name = result.rsplit('/')[-1].split('.')[0]

    val=json.load(open(result, 'r')) 

    scores = np.zeros(len(val))
    id = []
    for i in range(len(val)):
        if i+1 in coco_id_name_map: id.append(coco_id_name_map[i+1])
        else: id.append('None')
        scores[i] = val[i]['score']

    min_idx = heapq.nsmallest(30, range(len(scores)), scores.__getitem__)
    min_idx = np.array(min_idx)+1 

    plt.figure(figsize=(30,30))
    plt.title(f'{name}_COCO_test', fontsize=25)
    plt.xlabel('class', fontsize=25)
    plt.ylabel('score', fontsize=25)
    plt.xticks(rotation=-30)
    plt.bar(id, scores, color='b', align="center")
    plt.savefig(f'/home/dmsheng/demo/try/bar_{name}.png')  

def fiftyone_visual(dataset_dir, labels_path=None):
    import fiftyone as fo

    # The type of the dataset being imported
    dataset_type = fo.types.COCODetectionDataset  # for example

    
    if labels_path:
        # Import the COCO dataset
        dataset = fo.Dataset.from_dir(
            data_path=dataset_dir,
            dataset_type=dataset_type,
            labels_path=labels_path,
            shuffle=True
        )
    else:
        # Import the image dataset
        dataset = fo.Dataset.from_images_dir(dataset_dir)

    session = fo.launch_app(dataset, remote=True)
    session.wait()


data = '/home/dmsheng/demo/try/glide-text2im/OUTPUT/test/type_8/train2017/' # /home/dmsheng/demo/try/glide-text2im/experiments/center/train2017  /home/dmsheng/demo/try/glide-text2im/experiments/coco/train2017; /home/dmsheng/demo/obj_detection/PGNet/result
annFile = '/home/dmsheng/demo/try/glide-text2im/OUTPUT/test/type_8/annotations/instances_train2017.json'# '/home/dmsheng/demo/try/mmdetection/data/coco_5_aug_refine/annotations/instances_train2017.json' 
# annFile = None
fiftyone_visual(data, annFile) # 

# test()

# a = torch.tensor([0, 1, 1, 0])
# b = torch.tensor([0, 3, 2, 1, 1, 1])
# c = torch.tensor([6,7,8,9,10,5])
# print(a, b)
# idxs = torch.flatten(torch.nonzero(a))
# aug_labels = torch.zeros(11)
# for idx in iter(idxs):
#     # print(torch.where(b==idx)[0])
#     pos = torch.where(b==idx)[0]
#     print(pos, c[pos])
#     aug_labels[c[pos]] = 1
# print(aug_labels)