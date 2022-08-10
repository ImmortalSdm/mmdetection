from glob import glob
from mmdet.datasets.builder import PIPELINES
import json 
import cv2
import numpy as np
import os
from random import sample

class GetOutOfLoop(Exception):
    pass

coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
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

def intersection(s1,s2):
    area2 = (s2[3]-s2[1])*(s2[2]-s2[0])
    dx = min(s2[3], s1[3]) - max(s2[1], s1[1])
    dy = min(s2[2], s1[2]) - max(s2[0], s1[0])
    if (dx>=0) and (dy>=0):
        return dx*dy/area2
    else:
        return 0

@PIPELINES.register_module()
class Copy_Paste_Random:
    def __init__(self, fg_data_path='/mnt/home/syn4det/data/coco_ldm_80_1000/', P=0.5, N=1, scale_p=[0.2,2], care_overlap=True, mask_threshold=128):
        fg_data = {}
        for k,v in coco_id_name_map.items():
            fg_data[v] = glob(os.path.join(fg_data_path, v, '*'))
        self.fg_data = fg_data    
        self.P = P
        self.N = N
        self.scale_p = scale_p
        self.care_overlap = care_overlap
        self.mask_threshold = mask_threshold
        del fg_data

    def load_RGBA_BB(self,file_path,size):
        img_RGBA = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
        alpha = img_RGBA[...,3:]
        RGB = img_RGBA[...,:3]

        kernel  =  np.ones((3,3), np.uint8)
        mask_open  =  cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite('OUTPUT/test/mask_open.png', mask_open)
        kernel  =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_open  =  cv2.erode(mask_open, kernel = kernel)
        # cv2.imwrite('OUTPUT/test/mask_erode.png', mask_open.astype(np.uint8))
        num_labels, _, _, _  =  cv2.connectedComponentsWithStats(mask_open[:,:,0].astype(np.uint8), connectivity=8)
        if num_labels >= 5 or num_labels == 1:
            raise GetOutOfLoop
        # else:
        seg_mask  =  np.where(mask_open[:,:,0]>self.mask_threshold)
        y_min,y_max,x_min,x_max  =  np.min(seg_mask[0]), np.max(seg_mask[0]), np.min(seg_mask[1]), np.max(seg_mask[1])
        scale = size/max((y_max-y_min),(x_max-x_min))
        new_H = scale*(y_max-y_min)
        new_W = scale*(x_max-x_min)
        RGB = cv2.resize(RGB[y_min:y_max,x_min:x_max],(round(new_H),round(new_W)))
        alpha = cv2.resize(mask_open[y_min:y_max,x_min:x_max],(round(new_H),round(new_W)))

        alpha[alpha<self.mask_threshold] = 0
        alpha[alpha>=self.mask_threshold] = 1
        return RGB,alpha

    def try_add_syn(self,img,bboxes,labels,cls,care_overlap):
        cat = coco_id_name_map[cls]
        img_h,img_w = img.shape[:2]
        if len(self.fg_data[cat])==0:
            return 0
        sub_img = sample(self.fg_data[cat],1)[0]
        scales = []
        for lab,bbox in zip(labels,bboxes):
            if lab==cls:
                scales.append(max(bbox[3]-bbox[1], bbox[2]-bbox[0]))
        scale = np.mean(scales)*np.random.uniform(*self.scale_p)
        try:
            RGB,alpha = self.load_RGBA_BB(sub_img, float(scale))
        except:
            return 0
        ph,pw = RGB.shape[:2]
        dy = img_h-ph
        dx = img_w-pw
        if dy<=0 or dx<=0:
            return 0   
        dy = np.random.randint(dy)
        dx = np.random.randint(dx)
        if care_overlap:
            for bbox in bboxes:
                if intersection([dx, dy, dx+pw, dy+ph],bbox)>0.2:
                    return 0
        labels.append(cls)
        bboxes.append([dx, dy, dx+pw, dy+ph])
        img[dy:dy+ph,dx:dx+pw] = img[dy:dy+ph,dx:dx+pw]*(1-alpha[...,None])+RGB*alpha[...,None]
        return 1 

    def __call__(self,results):
        img = results['img']
        bboxes = results['gt_bboxes'].tolist()
        labels = results['gt_labels'].tolist()
        label_set = set(labels)
        assert len(labels)==len(bboxes)
        N = min(self.N,len(label_set))
        for i in sample(label_set,N):
            if np.random.rand()<=self.P:
                for _ in range(3):
                    if self.try_add_syn(img,bboxes,labels,i,self.care_overlap):
                        break
        results['gt_bboxes'] = np.array(bboxes,dtype = 'float32')
        results['gt_labels'] = np.array(labels)
        return results