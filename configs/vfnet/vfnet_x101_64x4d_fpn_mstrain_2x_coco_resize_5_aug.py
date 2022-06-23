_base_ = './vfnet_r50_fpn_mstrain_2x_coco.py'
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

dataset_type = 'CocoDataset'
classes = ('airplane', 'fire hydrant', 'stop sign', 'parking meter', 'bear')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_aug/annotations/instances_train2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_aug/train2017/'),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_aug/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_aug/val2017/'),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_aug/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_aug/val2017/'))