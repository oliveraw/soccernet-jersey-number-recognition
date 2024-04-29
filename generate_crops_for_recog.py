

# this script converts the coco annotations from roboflow into the format needed by mmocr:
# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html#ocrdataset
# https://mmocr.readthedocs.io/en/dev-1.x/migration/dataset.html?highlight=textrecogdataset#id3
# {
#   "metainfo":
#     {
#       "dataset_type": "TextRecogDataset",
#       "task_name": "textrecog",
#     },
#     "data_list":
#     [
#       {
#         "img_path": "test_img.jpg",
#         "instances":
#             [
#               {
#                 "text": "GRAND"
#               }
#             ]
#       }
#     ]
# }

import json
import os
import collections
import argparse
import cv2
import yaml

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--data_root', default='./soccernet-annotated', type=str, help='path to dataset, the dir with (test, train, valid) directories')
parser.add_argument('--output_dir', default='./soccernet-crops', type=str)
args = parser.parse_args()

splits = ['train', 'test']
ann_file = '_annotations.coco.json'
out_file = '_mmocr_annotations_textrecog.json'

# create the output directories
for split in splits:
    os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

METAINFO = dict(
    dataset_type="TextRecogDataset",
    task_name="textrecog",
)

def coco_to_recog_ann(category_id_to_name, ann):
    bbox = ann['bbox']      # coco format is [x1, y1, H, W], we need [x1, y1, x2, y2]
    return dict(
        bbox=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]],
        bbox_label=0,    # object category always 0 (text) in mmocr
        text=category_id_to_name[ann['category_id']],
        polygon=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]],
        ignore=False
    )

for split in splits:
    ann_path = os.path.join(args.data_root, split, ann_file)
    with open(ann_path) as f:
        ann_dict = json.load(f)
        annotations = ann_dict['annotations']   # list of {'id': 0, 'image_id': 0, 'category_id': 6, 'bbox': [18, 24, 17.11, 19.17], 'area': 327.999, 'segmentation': [], 'iscrowd': 0}
        categories = ann_dict['categories']     # list of {'id': 1, 'name': '1'}
        images = ann_dict['images']             # list of {'id': 0, 'license': 1, 'file_name': '1248_346_jpg.rf.4722277492e8e027f34145920e840e4d.jpg', 'height': 114, 'width': 37, 'date_captured': '2024-03-08T19:38:00+00:00'}
        print(split, "images:", len(images), "annotations:", len(annotations), "categories:", len(categories))

        categories = ['1', '1', '10', '11', '13', '14', '15', '16', '17', '2', '20', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '33', '34', '36', '4', '40', '44', '5', '50', '55', '6', '62', '7', '8', '9']
        # categories = sorted(categories)
        category_id_to_text = {i+1 : name for i, name in enumerate(categories)}
        image_id_to_filename = {x['id'] : x['file_name'] for x in images}

        data_list = []
        print(len(categories), categories)
        for ann in annotations:
            img_name = image_id_to_filename[ann['image_id']]
            img_path = os.path.join(args.data_root, split, img_name)
            img = cv2.imread(img_path)
            bbox = ann['bbox']
            crop = img[bbox[1]:round(bbox[1]+bbox[3]), bbox[0]:round(bbox[0]+bbox[2])]

            out_img_name = f"{img_name}_{ann['id']}.jpg"
            out_path = os.path.join(args.output_dir, split, out_img_name)
            cv2.imwrite(out_path, crop)

            text = category_id_to_text[ann['category_id']]
            data_list.append({
                "img_path": out_img_name, 
                "instances": [{ "text": text }]
            })
            print(out_img_name, text)
        out_dict = dict(
            metainfo=METAINFO,
            data_list=data_list
        )
        # print(json.dumps(out_dict, indent=4))
        out_path = os.path.join(args.output_dir, split, out_file)
        with open(out_path, 'w') as f:
            json.dump(out_dict, f)
        print(f"Writing {split} mmocr recog annotations to {out_path}")