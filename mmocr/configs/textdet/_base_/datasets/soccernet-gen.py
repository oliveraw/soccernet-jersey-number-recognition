# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html

soccernet_gen_textdet_train = dict(
    type='OCRDataset',
    data_root='data/soccernet-train/images',
    ann_file='gen-annotations-L-train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

soccernet_gen_textdet_val = dict(
    type='OCRDataset',
    data_root='data/soccernet-train/images',
    ann_file='gen-annotations-L-test.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

soccernet_gen_textdet_test = soccernet_gen_textdet_val

