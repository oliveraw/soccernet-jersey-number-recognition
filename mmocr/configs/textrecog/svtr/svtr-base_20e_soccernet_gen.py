_base_ = [
    'svtr-base_20e_st_mj.py',
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/soccernet-gen.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/svtr-base_20e_st_mj-ea500101.pth'

model = dict(
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)

# dataset settings
train_list = [
    _base_.cute80_textrecog_test,
    _base_.iiit5k_textrecog_train,
    _base_.svt_textrecog_train, 
    _base_.icdar2013_textrecog_train,
    _base_.icdar2015_textrecog_train,
    _base_.soccernet_gen_textrecog_train
]
test_list = [
    _base_.iiit5k_textrecog_test,
    _base_.svt_textrecog_test,
    _base_.svtp_textrecog_test,
    _base_.icdar2013_textrecog_test,
    _base_.icdar2015_textrecog_test,
    _base_.soccernet_gen_textrecog_test
]
# train_list = [_base_.soccernet_gen_textrecog_train]
# test_list = [_base_.soccernet_gen_textrecog_test]
val_list = test_list

val_evaluator = dict(dataset_prefixes=['IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15', "SoccernetGen"])
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_list,
        pipeline=_base_.test_pipeline))

test_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))