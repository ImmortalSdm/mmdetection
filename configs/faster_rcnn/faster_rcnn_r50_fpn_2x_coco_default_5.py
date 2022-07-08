_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
classes = ('airplane', 'fire hydrant', 'stop sign', 'parking meter', 'bear')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_default/annotations/instances_train2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_default/train2017/'),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_default/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_default/val2017/'),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/home/dmsheng/demo/try/mmdetection/data/coco_5_default/annotations/instances_val2017.json',
        img_prefix='/home/dmsheng/demo/try/mmdetection/data/coco_5_default/val2017/'))