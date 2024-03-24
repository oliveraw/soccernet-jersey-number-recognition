

# this script converts the coco annotations from roboflow into the format needed by mmocr:
# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html#ocrdataset
# {
#     "metainfo":
#     {
#       "dataset_type": "TextDetDataset",  # Options: TextDetDataset/TextRecogDataset/TextSpotterDataset
#       "task_name": "textdet",  #  Options: textdet/textspotter/textrecog
#       "category": [{"id": 0, "name": "text"}]  # Used in textdet/textspotter
#     },
#     "data_list":
#     [
#       {
#         "img_path": "test_img.jpg",
#         "height": 604,
#         "width": 640,
#         "instances":  # multiple instances in one image
#         [
#           {
#             "bbox": [0, 0, 10, 20],  # in textdet/textspotter, [x1, y1, x2, y2].
#             "bbox_label": 0,  # The object category, always 0 (text) in MMOCR
#             "polygon": [0, 0, 0, 10, 10, 20, 20, 0], # in textdet/textspotter. [x1, y1, x2, y2, ....]
#             "text": "mmocr",  # in textspotter/textrecog
#             "ignore": False # in textspotter/textdet. Whether to ignore this sample during training
#           },
#           #...
#         ],
#       }
#       #... multiple images
#     ]
# }

import json
import os
import collections

data_dir = "soccernet-annotated"
splits = ['train', 'test', 'valid']
ann_file = '_annotations.coco.json'
out_file = '_mmocr_annotations.json'

METAINFO = dict(
    dataset_type="TextDetDataset",
    task_name="textdet",
    category=[{"id": 0, "name": "text"}]
)

def coco_to_mmocr_ann(category_id_to_name, ann):
    bbox = ann['bbox']      # coco format is [x1, y1, H, W], we need [x1, y1, x2, y2]
    return dict(
        bbox=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]],
        bbox_label=0,    # object category always 0 (text) in mmocr
        text=category_id_to_name[ann['category_id']],
        polygon=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]],
        ignore=False
    )

for split in splits:
    ann_path = os.path.join(data_dir, split, ann_file)
    with open(ann_path) as f:
        ann_dict = json.load(f)
        annotations = ann_dict['annotations']   # list of {'id': 0, 'image_id': 0, 'category_id': 6, 'bbox': [18, 24, 17.11, 19.17], 'area': 327.999, 'segmentation': [], 'iscrowd': 0}
        categories = ann_dict['categories']     # list of {'id': 1, 'name': '1'}
        images = ann_dict['images']             # list of {'id': 0, 'license': 1, 'file_name': '1248_346_jpg.rf.4722277492e8e027f34145920e840e4d.jpg', 'height': 114, 'width': 37, 'date_captured': '2024-03-08T19:38:00+00:00'}
        print(split, "images:", len(images), "annotations:", len(annotations), "categories:", len(categories))

        category_id_to_name = {x['id'] : x['name'] for x in categories}

        image_id_to_annotations = collections.defaultdict(list)
        for ann in annotations:
            if len(ann['bbox']) == 0:       # skip the images with no bounding boxes
                continue
            mmocr_ann = coco_to_mmocr_ann(category_id_to_name, ann)
            image_id_to_annotations[ann['image_id']].append(mmocr_ann)

        full_data_list = []
        for image in images:
            if len(image_id_to_annotations[image['id']]) == 0:  # skip the images with no annotation
                continue
            image_path = image['file_name']
            h, w = image['height'], image['width']
            image_annotation = dict(
                img_path=image_path,
                height=h,
                width=w,
                instances=image_id_to_annotations[image['id']]
            )
            full_data_list.append(image_annotation)

        out_dict = dict(
            metainfo=METAINFO,
            data_list=full_data_list
        )
        # print(json.dumps(out_dict, indent=4))
        out_path = os.path.join(data_dir, split, out_file)
        with open(out_path, 'w') as f:
            json.dump(out_dict, f)
        print(f"Writing {split} mmocr annotations to {out_path}")