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
from sklearn.cluster import KMeans

from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.4, type=float)
parser.add_argument('--rec_threshold', default=0.95, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated_gen.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/soccernet-dbnetpp-genL/epoch_30.pth', type=str, help='weights for the finetuned detector')
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

# infer = MMOCRInferencer(det=args.det_config_path,
#     det_weights=args.det_weights_path,
#     rec="svtr-small",
#     device=device)
infer = MMOCRInferencer(det='dbnetpp',
    rec="svtr-small",
    device=device)

def get_crop(pred, frame_path):
    polygons = pred['det_polygons']
    det_scores = pred['det_scores']

    img = cv2.imread(frame_path)
    h, w, _ = img.shape

    # polygons, det_scores, bboxes, rec_texts all have the same length
    assert len(polygons) == len(det_scores)
    for i, det_score in enumerate(det_scores):
        img_name = "/".join(frame_path.split("/")[-2:])
        if det_score > args.det_threshold:
            polygons[i] = [round(x) for x in polygons[i]]
            bbox = poly2bbox(polygons[i])
            bbox = [round(x) for x in bbox]
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            return crop
    return None

for video_idx, (frame_paths, gt) in enumerate(dataset_to_use):
    # for debugging to skip most frames
    frame_paths = frame_paths[:10]

    # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output
    result = infer(frame_paths, out_dir=args.output_dir, save_vis=False)

    all_pixels = []
    # each pred corresponds to one frame of the video
    for frame_idx, pred in enumerate(result['predictions']):
        crop = get_crop(pred, frame_paths[frame_idx])
        if type(crop) == np.ndarray:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            pixels = crop.reshape(-1, 3)
            all_pixels.append(pixels)

    if len(all_pixels) == 0:
        continue

    all_pixels = np.concatenate(all_pixels, axis=0)
    all_colors = np.array(['#%02x%02x%02x' % tuple(rgb) for rgb in all_pixels])

    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(all_pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    cluster_colors = np.array(['#%02x%02x%02x' % tuple(rgb) for rgb in cluster_centers])
    values, counts = np.unique(kmeans.labels_, return_counts=True)
    max_cluster = np.argmax(counts)
    print("centers", cluster_centers, "max", max_cluster)

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    frame = cv2.imread(frame_paths[5])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax1.imshow(frame)     # hardcoded 10th frame to show

    ax2.scatter(all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2], c=all_colors)

    # plot cluster centers with the max cluster being a star
    for i, center in enumerate(cluster_centers):
        ax3.scatter(center[0], center[1], center[2], c=cluster_colors[i], marker="*" if i == max_cluster else "o")
        
    print("saving color histogram", f"colors/bbox_{video_idx}.jpg")
    os.makedirs("colors", exist_ok=True)
    plt.savefig(f"colors/bbox_{video_idx}.jpg")
    plt.close()
