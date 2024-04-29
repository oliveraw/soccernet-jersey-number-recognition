_base_ = [
    'svtr-base_20e_st_mj.py',
    '../_base_/datasets/soccernet.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/svtr-base_20e_st_mj-ea500101.pth'

model = dict(
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=2)

# dataset settings
train_list = [_base_.soccernet_textrecog_train]
test_list = [_base_.soccernet_textrecog_test]
val_list = test_list

val_evaluator = dict(dataset_prefixes=['Soccernet'])
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