BATCH_SIZE = 16
RESIZE = (
    400,
    400,
)
auto_scale_lr = dict(base_batch_size=8)
default_hooks = dict(
    checkpoint=dict(interval=20, type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=True,
        enable=True,
        interval=1,
        show=True,
        type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            True,
            True,
            True,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.4395,
            135.9471,
            84.0932,
        ],
        pad_size_divisor=32,
        std=[
            38.5333,
            38.7357,
            47.5903,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(
            epsilon_ratio=0.002, text_repr_type='quad',
            type='DBPostprocessor'),
        type='DBHead'),
    neck=dict(
        asf_cfg=dict(attention_type='ScaleChannelSpatial'),
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        lateral_channels=256,
        type='FPNC'),
    type='DBNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='Adam'), type='OptimWrapper')
pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        brightness=0.12549019607843137,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        args=[
            [
                'Fliplr',
                0.5,
            ],
            dict(cls='Affine', rotate=[
                -10,
                10,
            ]),
            [
                'Resize',
                [
                    0.5,
                    3.0,
                ],
            ],
        ],
        type='ImgAugWrapper'),
    dict(min_side_ratio=0.1, type='RandomCrop'),
    dict(keep_ratio=True, scale=(
        400,
        400,
    ), type='Resize'),
    dict(size=(
        400,
        400,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
        ),
        type='PackTextDetInputs'),
]
randomness = dict(seed=None)
resume = False
soccernet_test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='_mmocr_annotations.json',
        data_root='data/soccernet-annotated/test',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
soccernet_textdet_test = dict(
    ann_file='_mmocr_annotations.json',
    data_root='data/soccernet-annotated/test',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
soccernet_textdet_train = dict(
    ann_file='_mmocr_annotations.json',
    data_root='data/soccernet-annotated/train',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
soccernet_textdet_val = dict(
    ann_file='_mmocr_annotations.json',
    data_root='data/soccernet-annotated/valid',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
soccernet_train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='_mmocr_annotations.json',
        data_root='data/soccernet-annotated/train',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
soccernet_val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='_mmocr_annotations.json',
        data_root='data/soccernet-annotated/valid',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        datasets=[
            dict(
                ann_file='_mmocr_annotations.json',
                data_root='data/soccernet-annotated/test',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'instances',
                ),
                type='PackTextDetInputs'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
test_list = [
    dict(
        ann_file='_mmocr_annotations.json',
        data_root='data/soccernet-annotated/test',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
]
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'instances',
        ),
        type='PackTextDetInputs'),
]
train_cfg = dict(max_epochs=10, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        datasets=[
            dict(
                ann_file='_mmocr_annotations.json',
                data_root='data/soccernet-annotated/train',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                brightness=0.12549019607843137,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
            dict(
                args=[
                    [
                        'Fliplr',
                        0.5,
                    ],
                    dict(cls='Affine', rotate=[
                        -10,
                        10,
                    ]),
                    [
                        'Resize',
                        [
                            0.5,
                            3.0,
                        ],
                    ],
                ],
                type='ImgAugWrapper'),
            dict(min_side_ratio=0.1, type='RandomCrop'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(size=(
                640,
                640,
            ), type='Pad'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                ),
                type='PackTextDetInputs'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_list = [
    dict(
        ann_file='_mmocr_annotations.json',
        data_root='data/soccernet-annotated/train',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        brightness=0.12549019607843137,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        args=[
            [
                'Fliplr',
                0.5,
            ],
            dict(cls='Affine', rotate=[
                -10,
                10,
            ]),
            [
                'Resize',
                [
                    0.5,
                    3.0,
                ],
            ],
        ],
        type='ImgAugWrapper'),
    dict(min_side_ratio=0.1, type='RandomCrop'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(size=(
        640,
        640,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
        ),
        type='PackTextDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        datasets=[
            dict(
                ann_file='_mmocr_annotations.json',
                data_root='data/soccernet-annotated/valid',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'instances',
                ),
                type='PackTextDetInputs'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
val_list = [
    dict(
        ann_file='_mmocr_annotations.json',
        data_root='data/soccernet-annotated/valid',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualization = dict(
    draw_gt=False,
    draw_pred=True,
    enable=True,
    interval=1,
    show=True,
    type='VisualizationHook')
visualizer = dict(
    name='visualizer',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'soccernet-dbnetpp'
