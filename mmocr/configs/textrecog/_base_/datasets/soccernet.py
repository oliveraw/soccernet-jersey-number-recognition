# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html

soccernet_textrecog_train = dict(
    type='OCRDataset',
    data_root='data/soccernet-crops/train',
    ann_file='_mmocr_annotations_textrecog.json',
    pipeline=None)

soccernet_textrecog_test = dict(
    type='OCRDataset',
    data_root='data/soccernet-crops/test',
    ann_file='_mmocr_annotations_textrecog.json',
    pipeline=None)