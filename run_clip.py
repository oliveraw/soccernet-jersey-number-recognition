import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mmocr.apis import MMOCRInferencer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import cv2
import scipy
import numpy as np
import os
import json
import clip

from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat, generate_all_file_names

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.6, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/soccernet-dbnetpp/5-epochs/epoch_5.pth', type=str, help='weights for the finetuned detector')
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

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device}")

det_model = MMOCRInferencer(det=args.det_config_path, det_weights=args.det_weights_path, device=device)
clip_model, preprocess = clip.load('ViT-B/32', device)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

train_dataset = soccernet_dataset_flat(args.data_path, "train", preprocess)
test_dataset = soccernet_dataset_flat(args.data_path, "test", preprocess)
logger.info(f"Num images in train dataset: {len(train_dataset)}")
logger.info(f"Num images in test dataset: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

def visualize_batch(batch, probs, path):
    caption = ""
    for idx, prob in enumerate(probs):
        if idx % 4 == 0 and idx != 0:
            caption += '\n'
        caption += str(prob) + " | "

    grid = make_grid(batch, nrow=4, padding=2)
    grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.title(caption)
    plt.imshow(grid_np)
    plt.axis('off')
    plt.savefig(path)

text = clip.tokenize(["a soccer player with a visible jersey number", "a soccer player with no visible jersey number"]).to(device)
for idx, (imgs, gt, frame_paths) in enumerate(train_dataloader):
    result = det_model(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)

    for p_idx, pred in enumerate(result['predictions']):
        det_scores = pred['det_scores']
    visualize_batch(imgs, [pred['det_scores'] for pred in result['predictions']], f"det_vis_{idx}.png")

    # with torch.no_grad():
    #     imgs, gt = imgs.to(device), gt.to(device)

    #     logits_per_image, logits_per_text = model(imgs, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #     print(probs)

    #     visualize_batch(imgs, probs, f"clip_vis_{idx}.png")

    #     if idx == 9:
    #         break

################################### copied ######################################
# correct = []
# for video_idx, (frame_paths, gt) in enumerate(test_dataset):
#     # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output

#     # for debugging to skip most frames
#     # frame_paths = frame_paths[:2]

#     result = infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)
#     predictions = []
#     for idx, pred in enumerate(result['predictions']):
#         det_scores = pred['det_scores']
#         rec_scores = pred['rec_scores']
#         rec_texts = pred['rec_texts']
#         # print(idx, det_scores, rec_texts)
#         predictions.extend(list(zip(det_scores, rec_scores, rec_texts)))

#         # save the images which were over the detection threshold
#         if args.save_vis:
#             over_threshold = len(det_scores) != 0 and any(i >= args.det_threshold for i in det_scores)
#             if over_threshold:
#                 filename = frame_paths[idx].split('/')[-1]
#                 plt.figure()
#                 plt.title(det_scores)
#                 plt.imshow(result['visualization'][0])
#                 plt.savefig(f"{args.output_dir}/soccernet-{os.getenv('SLURM_JOB_ID')}/vis/{filename}")
#                 logger.debug(f"Saving figure {filename}")

#     confident_numbers = []
#     # filter non numeric predictions, take only predictions with det confidence above threshold
#     for i, (det_score, rec_score, rec_text) in enumerate(predictions):
#         if det_score > args.det_threshold and rec_text.isnumeric():
#             confident_numbers.append(int(rec_text))

#     confident_numbers = np.array(confident_numbers)
#     final_prediction = scipy.stats.mode(confident_numbers, axis=None, keepdims=False)[0]
#     if np.isnan(final_prediction):
#         final_prediction = -1
#     correct.append(final_prediction == gt)

#     logger.info(f"Video: {video_idx}, Prediction: {final_prediction}, Ground truth: {gt} Correct?: {final_prediction == gt}")
#     for pred in predictions:
#         logger.debug(pred)

# logger.info(f"Final Accuracy: {sum(correct)}/{len(correct)} = {sum(correct) / len(correct)}")