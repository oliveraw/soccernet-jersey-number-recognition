_base_ = [
    'svtr-base_20e_st_mj.py',
    '../_base_/datasets/toy_data.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/svtr-base_20e_st_mj-ea500101.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=3)

# dataset settings
train_list = [_base_.toy_rec_train]
test_list = [_base_.toy_rec_test]
val_list = test_list

val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
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
    drop_last=False,
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
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))