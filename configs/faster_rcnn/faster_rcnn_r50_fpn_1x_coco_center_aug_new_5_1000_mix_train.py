_base_ = [
    '../_base_/models/my_faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'MixedDataset'
classes = ('airplane', 'fire hydrant', 'stop sign', 'parking meter', 'bear')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MYLoadAnnotations', with_bbox=True, with_aug=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='MYDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'aug_labels']),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_center_aug_new/annotations/instances_train2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_center_aug_new/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_center_aug_new/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_center_aug_new/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_center_aug_new/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_center_aug_new/val2017/',
        pipeline=test_pipeline))