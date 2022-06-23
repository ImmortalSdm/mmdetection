# 在mmdetection的根目录下运行，如果报错：没有那个参数，就把create_mm_config中那个参数赋值给注释掉。生成配置文件后，直接修改配置文件就可以了。
import os
from mmcv import Config

#################################  下边是要修改的内容   ####################################

root_path = os.getcwd()
model_name = 'vfnet_x101_64x4d'  # 改成自己要使用的模型名字
work_dir = os.path.join(root_path, "work_dirs", model_name)  # 训练过程中，保存日志权重文件的路径，。
baseline_cfg_path = os.path.join('configs', 'vfnet', 'vfnet_r50_fpn_mstrain_2x_coco.py')
# 改成自己要使用的模型的配置文件路径
save_cfg_path = os.path.join(work_dir, 'config.py')  # 生成的配置文件保存的路径

train_data_images = os.path.join(root_path, 'data', 'coco', 'train', 'images')  # 改成自己训练集图片的目录。
val_data_images = os.path.join(root_path, 'data', 'coco', 'train', 'images')  # 改成自己验证集图片的目录。
test_data_images = os.path.join(root_path, 'data', 'coco', 'val', 'images')  # 改成自己测试集图片的目

train_ann_file = os.path.join(root_path, 'data', 'coco', 'train', 'annotations', 'new_train.json')  # 修改为自己的数据集的训练集json
val_ann_file = os.path.join(root_path, 'data', 'coco', 'train', 'annotations', 'new_val.json')  # 修改为自己的数据集的验证集json
test_ann_file = os.path.join(root_path, 'data', 'coco', 'val', 'annotations', 'new_test.json')  # 修改为自己的数据集的验证集json录。

# 去找个网址里找你对应的模型的网址: https://github.com/open-mmlab/mmdetection/blob/master/README_zh-CN.md
# load_from = os.path.join(work_dir, 'checkpoint.pth')  # 修改成自己的checkpoint.pth路径

# File config
num_classes = 5  # 改成自己的类别数。
classes = ('1', '2', '3', '4', '5')  # 改成自己的类别，如果只有一个类别的话，要写成这样定义为元组: classes = ('1', )

###############  下边一些参数包含不全，可以在生成的配置文件中再对其他参数进行修改    #####################

# Train config              # 根据自己的需求对下面进行配置
gpu_ids = 4,5  # 改成自己要用的gpu
gpu_num = 2
total_epochs = 20  # 改成自己想训练的总epoch数
batch_size = 2 ** 2  # 根据自己的显存，改成合适数值，建议是2的倍数。
num_worker = 1  # 比batch_size小，就行
log_interval = 300  # 日志打印的间隔
checkpoint_interval = 7  # 权重文件保存的间隔
lr = 0.02 * batch_size * gpu_num / 16  # 学习率
ratios = [0.5, 1.0, 2.0]
strides = [4, 8, 16, 32, 64]

cfg = Config.fromfile(baseline_cfg_path)

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

cfg.work_dir = work_dir
print("Save config dir:", work_dir)

# swin和mmdetection的训练集配置不在一个地方，那个不报错用哪个：
cfg.classes = classes
# mmdetection用这个：
cfg.data.train.img_prefix = train_data_images
cfg.data.train.classes = classes
cfg.data.train.ann_file = train_ann_file
# swin用这个，注释上边那个
# cfg.data.train.dataset.img_prefix = train_data_images
# cfg.data.train.dataset.classes = classes
# cfg.data.train.dataset.ann_file = train_ann_file

cfg.data.val.img_prefix = val_data_images
cfg.data.val.classes = classes
cfg.data.val.ann_file = val_ann_file

cfg.data.test.img_prefix = test_data_images
cfg.data.test.classes = classes
cfg.data.test.ann_file = test_ann_file

cfg.data.samples_per_gpu = batch_size
cfg.data.workers_per_gpu = num_worker
cfg.log_config.interval = log_interval

# 有些配置文件num_classes可能不在这个地方，生成之后去配置文件里搜索一下，看看都修改了没
for head in cfg.model.roi_head.bbox_head:
    head.num_classes = num_classes
if "mask_head" in cfg.model.roi_head:
    cfg.model.roi_head.mask_head.num_classes = num_classes

# cfg.load_from = load_from
cfg.runner.max_epochs = total_epochs
cfg.total_epochs = total_epochs
cfg.optimizer.lr = lr
cfg.checkpoint_config.interval = checkpoint_interval
cfg.model.rpn_head.anchor_generator.ratios = ratios
cfg.model.rpn_head.anchor_generator.strides = strides

cfg.dump(save_cfg_path)
print(save_cfg_path)
print("—" * 50)
print(f'CONFIG:\n{cfg.pretty_text}')
print("—" * 50)
