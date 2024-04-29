<<<<<<< HEAD
# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html
=======
# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/transforms.html
>>>>>>> b11a05adc516705c385bfdfd9080020049f79c4a

soccernet_textdet_train = dict(
    type='OCRDataset',
    data_root='data/soccernet-annotated/train',
    ann_file='_mmocr_annotations.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

soccernet_textdet_val = dict(
    type='OCRDataset',
    data_root='data/soccernet-annotated/valid',
    ann_file='_mmocr_annotations.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

soccernet_textdet_test = dict(
    type='OCRDataset',
    data_root='data/soccernet-annotated/test',
    ann_file='_mmocr_annotations.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
