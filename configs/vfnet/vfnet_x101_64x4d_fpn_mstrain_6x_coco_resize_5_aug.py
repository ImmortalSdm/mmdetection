_base_ = './vfnet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
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
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
# Use RepeatDataset to speed up training
dataset_type = 'CocoDataset'
classes = ('airplane', 'fire hydrant', 'stop sign', 'parking meter', 'bear')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_resize_aug/annotations/instances_train2017.json',
            img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_resize_aug/train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_resize_aug/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_resize_aug/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_resize_aug/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_resize_aug/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
