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

from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat, generate_all_file_names

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.6, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/soccernet-dbnetpp/epoch_10.pth', type=str, help='weights for the finetuned detector')
args = parser.parse_args()

# toggle between INFO, DEBUG
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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

infer = MMOCRInferencer(det=args.det_config_path, det_weights=args.det_weights_path, rec='SAR', device=device)

correct = []
for video_idx, (frame_paths, gt) in enumerate(train_dataset):
    # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output

    result = infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)
    predictions = []
    for idx, pred in enumerate(result['predictions']):
        det_scores = pred['det_scores']
        rec_scores = pred['rec_scores']
        rec_texts = pred['rec_texts']
        # print(idx, det_scores, rec_texts)

        # filter non numeric predictions, take only predictions with det confidence above threshold
        for i, score in enumerate(det_scores):
            text = rec_texts[i]
            if score > args.det_threshold and text.isnumeric():
                predictions.append(int(text))

        # save the images which were over the detection threshold
        # over_threshold = len(det_scores) != 0 and any(i >= args.det_threshold for i in det_scores)
        # if over_threshold:
        #     filename = frame_paths[idx].split('/')[-1]
        #     plt.figure()
        #     plt.title(det_scores)
        #     plt.imshow(result['visualization'][0])
        #     plt.savefig(f"{args.output_dir}/{filename}")
        #     logger.debug(f"Saving figure {filename}")

    predictions = np.array(predictions)
    final_prediction = scipy.stats.mode(predictions, axis=None, keepdims=False)[0]
    if np.isnan(final_prediction):
        final_prediction = -1
    print("Video:", video_idx, "Prediction:", final_prediction, "Ground truth:", gt, "Correct?:", final_prediction == gt, predictions)
    correct.append(final_prediction == gt)

print(f"Final Accuracy: {count(correct)}/{len(correct)} = {count(correct) / len(correct)}")