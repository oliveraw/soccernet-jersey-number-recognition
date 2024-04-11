# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html

soccernet_gen_textrecog_train = dict(
    type='OCRDataset',
    data_root='data/soccernet-crops-gen-L/train',
    ann_file='gen-annotations-L.json',
    pipeline=None)

soccernet_gen_textrecog_test = dict(
    type='OCRDataset',
    data_root='data/soccernet-crops-gen-L/test',
    ann_file='gen-annotations-L.json',
    pipeline=None)