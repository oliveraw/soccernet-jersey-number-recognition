_base_ = [
    '_base_fcenet_resnet50_fpn_jo.py',
    # '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/soccernet.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015/fcenet_resnet50_fpn_1500e_icdar2015_20220826_140941-167d9042.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=5)

train_list = [_base_.soccernet_textdet_train]
val_list = [_base_.soccernet_textdet_val]
test_list = [_base_.soccernet_textdet_test]

batch_size = 8

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_list,
        pipeline=_base_.test_pipeline))

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

auto_scale_lr = dict(base_batch_size=batch_size)

# _base_.model.data_preprocessor=dict(
#     type='TextDetDataPreprocessor',
#     mean=[127.4395, 135.9471,  84.0932],
#     std=[38.5333, 38.7357, 47.5903],
#     bgr_to_rgb=True,
#     pad_size_divisor=32)

# train_dataloader = dict(
#     batch_size=8,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=icdar2015_textdet_train)

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=icdar2015_textdet_test)

# test_dataloader = val_dataloader

