_base_ = [
    '_base_fcenet_resnet50_fpn.py',
    # '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/soccernet.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_10e.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015/fcenet_resnet50_fpn_1500e_icdar2015_20220826_140941-167d9042.pth'


# optim_wrapper = dict(optimizer=dict(lr=1e-3, weight_decay=5e-4))
# train_cfg = dict(max_epochs=1500)
# learning policy
# param_scheduler = [
#     dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
# ]

# dataset settings
# icdar2015_textdet_train = _base_.icdar2015_textdet_train
# icdar2015_textdet_test = _base_.icdar2015_textdet_test
# icdar2015_textdet_train.pipeline = _base_.train_pipeline
# icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_list = [_base_.soccernet_textdet_train]
val_list = [_base_.soccernet_textdet_val]
test_list = [_base_.soccernet_textdet_test]

# train_pipeline = [
#     dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
#     dict(
#         type='LoadOCRAnnotations',
#         with_bbox=True,
#         with_polygon=True,
#         with_label=True,
#     ),
#     dict(
#         type='TorchVisionWrapper',
#         op='ColorJitter',
#         brightness=32.0 / 255,
#         saturation=0.5),
#     dict(
#         type='ImgAugWrapper',
#         args=[['Fliplr', 0.5],
#               dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
#     dict(type='RandomCrop', min_side_ratio=0.4),
#     dict(type='Resize', scale=(640, 640), keep_ratio=True),
#     dict(type='Pad', size=(640, 640)),
#     dict(
#         type='PackTextDetInputs',
#         meta_keys=('img_path', 'ori_shape', 'img_shape'))
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
#     dict(type='Resize', scale=(640, 640), keep_ratio=True),
#     dict(type='Pad', size=(640, 640)),
#     dict(
#         type='LoadOCRAnnotations',
#         with_polygon=True,
#         with_bbox=True,
#         with_label=True,
#     ),
#     dict(
#         type='PackTextDetInputs',
#         meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
#                    'instances'))
# ]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_list,
        pipeline=_base_.test_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

auto_scale_lr = dict(base_batch_size=8)

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

auto_scale_lr = dict(base_batch_size=8)
