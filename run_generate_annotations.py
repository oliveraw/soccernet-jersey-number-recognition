import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mmocr.apis import MMOCRInferencer
from mmocr.utils import poly2bbox
from torch.utils.data import DataLoader
import cv2
import scipy
import numpy as np
import os
import json

from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.6, type=float)
parser.add_argument('--rec_threshold', default=0.95, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated_gen.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/soccernet-dbnetpp/epoch_30.pth', type=str, help='weights for the finetuned detector')
parser.add_argument('--rec_config_path', default='mmocr/configs/textrecog/svtr/svtr-base_20e_soccernet_gen.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--rec_weights_path', default='mmocr/soccernet-svtr/epoch_20.pth', type=str, help='weights for the finetuned recognizer')
parser.add_argument('--img_output_dir', default="soccernet-crops-gen-L", type=str)
parser.add_argument('--save_vis', action='store_true')
args = parser.parse_args()

# create the output and vis directories
# os.makedirs(f"{args.output_dir}/soccernet-{os.getenv('SLURM_JOB_ID')}/vis", exist_ok=True)

# toggle between INFO, DEBUG
logging.basicConfig(format='%(asctime)s %(message)s', 
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device}")

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

train_dataset = soccernet_dataset(args.data_path, "train")
test_dataset = soccernet_dataset(args.data_path, "test")
logger.info(f"Num videos in train dataset: {len(train_dataset)}")
logger.info(f"Num videos in test dataset: {len(test_dataset)}")

dataset_to_use = train_dataset

infer = MMOCRInferencer(det=args.det_config_path,
    det_weights=args.det_weights_path,
    rec="svtr-small",
    device=device)

def convert_pred_to_det_rec_annotations(pred, gt, frame_path):
    polygons = pred['det_polygons']
    det_scores = pred['det_scores']
    rec_texts = pred['rec_texts']

    img = cv2.imread(frame_path)
    h, w, _ = img.shape

    # polygons, det_scores, bboxes, rec_texts all have the same length
    assert len(polygons) == len(det_scores) == len(rec_texts)
    for i, det_score in enumerate(det_scores):
        img_name = "/".join(frame_path.split("/")[-2:])
        if det_score > args.det_threshold and rec_texts[i] == str(gt):
            # 70/30 train/test split
            if random.random() < 0.7:
                split = "train"
            else:
                split = "test"

            polygons[i] = [round(x) for x in polygons[i]]
            bbox = poly2bbox(polygons[i])
            bbox = [round(x) for x in bbox]
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            crop_path = os.path.join(args.img_output_dir, split, img_name)
            os.makedirs(os.path.join(args.img_output_dir, split, img_name.split('/')[0]), exist_ok=True)
            cv2.imwrite(crop_path, crop)

            det_item = {
                "img_path": img_name,
                "height": h,
                "width": w,
                "instances": [
                    {
                        "polygon": polygons[i],
                        "bbox": bbox,
                        "bbox_label": 0,
                        "ignore": False,
                    }
                ]
            }
            rec_item = {
                "img_path": img_name,
                "instances": [
                    {
                        "text": str(gt)
                    }
                ]
            }
            return det_item, rec_item, split
    return None, None, None

det_data_train = []
det_data_test = []
rec_data_train = []
rec_data_test = []
for video_idx, (frame_paths, gt) in enumerate(dataset_to_use):
    # for debugging to skip most frames
    # frame_paths = frame_paths[:2]

    # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output
    result = infer(frame_paths, out_dir=args.output_dir, save_vis=False)

    # each pred corresponds to one frame of the video
    for frame_idx, pred in enumerate(result['predictions']):
        det_item, rec_item, split = convert_pred_to_det_rec_annotations(pred, gt, frame_paths[frame_idx])
        if det_item != None and rec_item != None:
            # print(json.dumps(det_item, indent=4))
            # print(json.dumps(rec_item, indent=4))
            if split == "train":
                det_data_train.append(det_item)
                rec_data_train.append(rec_item)
            else:
                det_data_test.append(det_item)
                rec_data_test.append(rec_item)


det_meta = {
    "dataset_type": "TextDetDataset",
    "task_name": "textdet",
    "category": [{"id": 0, "name": "text"}]
}
rec_meta = {
    "dataset_type": "TextRecogDataset",
    "task_name": "textrecog"
}

det_train_annotations = {
    "metainfo": det_meta,
    "data_list": det_data_train
}
rec_train_annotations = {
    "metainfo": rec_meta,
    "data_list": rec_data_train
}
det_test_annotations = {
    "metainfo": det_meta,
    "data_list": det_data_test
}
rec_test_annotations = {
    "metainfo": rec_meta,
    "data_list": rec_data_test
}

with open("data/train/images/gen-annotations-L-train.json", "w") as f:
    json.dump(det_train_annotations, f)
with open("data/train/images/gen-annotations-L-test.json", "w") as f:
    json.dump(det_test_annotations, f)

with open(f"{args.img_output_dir}/train/gen-annotations-L.json", "w") as f:
    json.dump(rec_train_annotations, f)
with open(f"{args.img_output_dir}/test/gen-annotations-L.json", "w") as f:
    json.dump(rec_test_annotations, f)
