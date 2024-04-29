# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html

soccernet_gen_textdet_train = dict(
    type='OCRDataset',
<<<<<<< HEAD
    data_root='data/soccernet-train/images',
    ann_file='gen-annotations-L-train.json',
=======
    data_root='data/soccernet-train-det-gen',
    ann_file='gen-annotations-L.json',
>>>>>>> b11a05adc516705c385bfdfd9080020049f79c4a
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

soccernet_gen_textdet_val = dict(
    type='OCRDataset',
<<<<<<< HEAD
    data_root='data/soccernet-train/images',
    ann_file='gen-annotations-L-test.json',
=======
    data_root='data/soccernet-test-det-gen',
    ann_file='gen-annotations-L.json',
>>>>>>> b11a05adc516705c385bfdfd9080020049f79c4a
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

soccernet_gen_textdet_test = soccernet_gen_textdet_val
<<<<<<< HEAD

=======
>>>>>>> b11a05adc516705c385bfdfd9080020049f79c4a
