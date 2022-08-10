_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush')

dataset_A_train = dict(
    type='RepeatDataset',
    times=5,
    dataset=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/glide-text2im/experiments/center_new_1000/annotations/instances_train2017.json',
        img_prefix='/home/dmsheng/demo/try/glide-text2im/experiments/center_new_1000/train2017/',
        pipeline=train_pipeline
    )
)

dataset_B_train = dict(
    type=dataset_type,
    # 将类别名字添加至 `classes` 字段中
    classes=classes,
    ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_default_all/annotations/instances_train2017.json',
    img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_default_all/train2017/',
    pipeline=train_pipeline
)
dataset_B_val = dict(
    type=dataset_type,
    classes=classes,
    ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_default_all/annotations/instances_val2017.json',
    img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_default_all/val2017/',
    pipeline=test_pipeline
)
dataset_B_test = dict(
    type=dataset_type,
    # 将类别名字添加至 `classes` 字段中
    classes=classes,
    ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_default_all/annotations/instances_val2017.json',
    img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_default_all/val2017/',
    pipeline=test_pipeline
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_B_val,
    test = dataset_B_test
)