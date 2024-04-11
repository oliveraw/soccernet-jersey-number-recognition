import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mmocr.apis import MMOCRInferencer
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
parser.add_argument('--det_threshold', default=0.6, type=float)
parser.add_argument('--rec_threshold', default=0.9, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated_gen.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/soccernet-dbnetpp-genL/epoch_30.pth', type=str, help='weights for the finetuned detector')
parser.add_argument('--rec_config_path', default='mmocr/configs/textrecog/svtr/svtr-base_20e_soccernet_gen.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--rec_weights_path', default='mmocr/soccernet-svtr/epoch_20.pth', type=str, help='weights for the finetuned recognizer')
parser.add_argument('--save_vis', action='store_true')
args = parser.parse_args()

# create the output and vis directories
os.makedirs(f"{args.output_dir}/soccernet-{os.getenv('SLURM_JOB_ID')}/vis", exist_ok=True)

# toggle between INFO, DEBUG
logfile = f"{args.output_dir}/soccernet-{os.getenv('SLURM_JOB_ID')}/output.log"
logging.basicConfig(filename=logfile,
    format='%(asctime)s %(message)s', 
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

# det = MMOCRInferencer(det=args.det_config_path,
#     det_weights=args.det_weights_path,
#     device=device)
# rec = MMOCRInferencer(rec=args.rec_config_path,
#     rec_weights=args.rec_weights_path,
#     device=device)

# infer = MMOCRInferencer(det=args.det_config_path,
#     det_weights=args.det_weights_path,
#     rec="svtr-small",
#     device=device)

correct = []
for video_idx, (frame_paths, gt) in enumerate(test_dataset):
    # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output

    # for debugging to skip most frames
    # frame_paths = frame_paths[:4]

    # FIX THIS - predicts non soccerball as soccerball sometimes
    HALF_CROP_SIZE = 10      # take crop from center of each image
    max_shape = 0
    all_pixels = []
    for path in frame_paths:
        img = cv2.imread(path)
        max_shape = max(max_shape, img.shape[0])
        max_shape = max(max_shape, img.shape[1])

        # taking center crop
        c_h, c_w = img.shape[0] // 2, img.shape[1] // 2
        crop = img[c_h-HALF_CROP_SIZE : c_h+HALF_CROP_SIZE, c_w-HALF_CROP_SIZE:c_w+HALF_CROP_SIZE]

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) 
        pixels = crop_rgb.reshape(-1, 3)
        all_pixels.append(pixels)

        # https://stackoverflow.com/questions/12182891/plot-image-color-histogram-using-matplotlib
        # color = ('b', 'g', 'r')
        # for i,col in enumerate(color_record.keys()):
        #     histr = cv2.calcHist([img],[i],None,[256],[0,256])
        #     color_record[col] += histr
    
    # print([x.shape for x in all_pixels])
    all_pixels = np.concatenate(all_pixels, axis=0)
    all_colors = np.array(['#%02x%02x%02x' % tuple(rgb) for rgb in all_pixels])
    # choose = np.random.choice(all_pixels.shape[0], 100000, replace=False)
    # all_pixels = all_pixels[choose, :]
    # all_colors = all_colors[choose]
    print(all_pixels.shape, all_colors.shape)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(all_pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    cluster_colors = np.array(['#%02x%02x%02x' % tuple(rgb) for rgb in cluster_centers])
    print("centers", cluster_centers)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.scatter(all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2], c=all_colors)
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c=cluster_colors)
    print("saving color histogram", f"colors/center_crop_{path.split('/')[-1]}")
    os.makedirs("colors", exist_ok=True)
    plt.savefig(f"colors/center_crop_{path.split('/')[-1]}")
    plt.close()
    
#     if max_shape < 50:
#         final_prediction = -1
#         logger.info(f"Video: {video_idx}, soccer ball shortcut prediction")
#     else:
#         # det_result = det(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)
#         # rec_result = rec(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)
#         # print(det_result['predictions'][0])
#         # assert False

#         result = infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)

#         predictions = []
#         for idx, pred in enumerate(result['predictions']):
#             det_scores = pred['det_scores']
#             rec_scores = pred['rec_scores']
#             rec_texts = pred['rec_texts']
#             # print(idx, det_scores, rec_texts)
#             predictions.extend(list(zip(det_scores, rec_scores, rec_texts)))

#             # save the images which were over the detection threshold
#             if args.save_vis:
#                 over_threshold = len(det_scores) != 0 and any(i >= args.det_threshold for i in det_scores)
#                 if over_threshold:
#                     filename = frame_paths[idx].split('/')[-1]
#                     plt.figure()
#                     plt.title(det_scores)
#                     plt.imshow(result['visualization'][0])
#                     plt.savefig(f"{args.output_dir}/soccernet-{os.getenv('SLURM_JOB_ID')}/vis/{filename}")
#                     logger.debug(f"Saving figure {filename}")

#         confident_numbers = []
#         # filter non numeric predictions, take only predictions with det confidence above threshold
#         for i, (det_score, rec_score, rec_text) in enumerate(predictions):
#             if det_score > args.det_threshold and rec_score > args.rec_threshold and rec_text.isnumeric():
#                 confident_numbers.append(int(rec_text))

#         confident_numbers = np.array(confident_numbers)
#         final_prediction = scipy.stats.mode(confident_numbers, axis=None, keepdims=False)[0]
#         if np.isnan(final_prediction):
#             final_prediction = -1

#     correct.append(final_prediction == gt)

#     logger.info(f"Video: {video_idx}, Prediction: {final_prediction}, Ground truth: {gt} Correct?: {final_prediction == gt}")
#     for pred in predictions:
#         logger.debug(pred)

# logger.info(f"Final Accuracy: {sum(correct)}/{len(correct)} = {sum(correct) / len(correct)}")