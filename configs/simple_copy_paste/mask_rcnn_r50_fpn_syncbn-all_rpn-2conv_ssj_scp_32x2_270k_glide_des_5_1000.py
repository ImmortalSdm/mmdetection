_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed.
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)
model = dict(
    backbone=dict(frozen_stages=-1, norm_eval=False, norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),  # leads to 0.1+ mAP
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=head_norm_cfg),
        mask_head=dict(norm_cfg=head_norm_cfg)))


# Use RepeatDataset to speed up training
dataset_type = 'CocoDataset'
classes = ('airplane', 'fire hydrant', 'stop sign', 'parking meter', 'bear')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

file_client_args = dict(backend='disk')

# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
load_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.8, 1.25),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size=image_size),
]
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
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
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=load_pipeline),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_ldm_t2i_w_5_3000/annotations/instances_train2017.json',
            img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_ldm_t2i_w_5_3000/train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_ldm_t2i_w_5_3000/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_ldm_t2i_w_5_3000/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_ldm_t2i_w_5_3000/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_ldm_t2i_w_5_3000/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)