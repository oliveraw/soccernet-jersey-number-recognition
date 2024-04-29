
import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mmocr.apis import MMOCRInferencer, TextDetInferencer, TextRecInferencer
from torch.utils.data import DataLoader
import cv2
import scipy
import numpy as np
import os, shutil
import json
from tqdm import tqdm
from collections import defaultdict
from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat, generate_all_file_names

#%%
def longest_continuous_numbers(numbers):
    seq_lengths = defaultdict(int)    
    current_streak = 1
    prev_num = None
    for num in numbers:
        if prev_num is None or num == prev_num:
            current_streak += 1
        else:
            seq_lengths[prev_num] = max(seq_lengths[prev_num], current_streak)
            current_streak = 1
        prev_num = num
    seq_lengths[prev_num] = max(seq_lengths[prev_num], current_streak)
    
    # Find the top 3 longest continuous sequences
    top_3 = sorted(seq_lengths.items(), key=lambda x: x[1], reverse=True)[:3]
    return top_3

def get_weighted_most_frequent_number(numbers):
    
    weights = {}
    # Assign weights based on the given rules
    for num in list(set(numbers)):
        if 10 <= num < 100:
            weights[num] = weights.get(num, 0) + 1.05
        else:
            weights[num] = weights.get(num, 0) + 1
    
    # Find the longest continuous sequence of the same number
    top3 = longest_continuous_numbers(numbers)
    for num, length in top3:
        weights[num] *= (1+ 0.01 * length)
    
    # Find the number with the highest weight
    most_frequent_num = max(weights, key=weights.get)
    return most_frequent_num


#%%

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.95, type=float)
parser.add_argument('--rec_threshold', default=0.8, type=float)
parser.add_argument('--restart_inference', action='store_true')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_soccernetannotated.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/jocelyn-output/fce_epoch_10.pth', type=str, help='weights for the finetuned detector')
parser.add_argument('--rec_config_path', default='mmocr/soccernet-svtr-genL-combined/svtr-small_20e_soccernet_gen.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--rec_weights_path', default='mmocr/soccernet-svtr-genL-combined/epoch_10.pth', type=str, help='weights for the finetuned recognitor')

args = parser.parse_args()

# toggle between INFO, DEBUG
logfile = f"logs/soccernet-{os.getenv('SLURM_JOB_ID')}-info.log"
logging.basicConfig(filename=logfile,
    format='%(asctime)s %(message)s', 
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logger.info(f"Using {device}")

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# train_dataset = soccernet_dataset(args.data_path, "train")
test_dataset = soccernet_dataset(args.data_path, "test")
# logger.info(f"Num videos in train dataset: {len(train_dataset)}")
logger.info(f"Num videos in test dataset: {len(test_dataset)}")
# debug_dataset = soccernet_dataset(args.data_path, "debug")
# logger.info(f"Num videos in test dataset: {len(debug_dataset)}")

#%%

det_infer = TextDetInferencer(model=args.det_config_path, weights=args.det_weights_path, device=device)
recognizer_names = ['svtr-small', 'NRTR', 'SATRN_sm', 'SAR', 'ABINet']
recognizers = [TextRecInferencer(model=name) for name in recognizer_names]

# rec_infer = TextRecInferencer(model=args.rec_config_path, weights=args.rec_weights_path, device=device)
# rec_infer = TextRecInferencer(model='svtr-small',device=device)

# logger.info(f"Detector: Finetuned FCENet, Recognizer: SVTR-small")

correct = []
weighted_correct = []

checkpoint_path = os.path.join(args.output_dir, "preds.json")
if not args.restart_inference and os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'r') as f:
        output_json = json.load(f)
else:
    output_json = {}
already_ran = len(output_json.keys())
idx_to_use = range(already_ran, len(test_dataset))
subset = torch.utils.data.Subset(test_dataset, idx_to_use)

for video_idx, (frame_paths, gt) in enumerate(subset, start=already_ran):
    # for debugging, comment as needed
    # frame_paths = frame_paths[:1]

    predictions = []
    result = det_infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=False)
    idx_to_rec = []
    cropped_imgs = []

    shapes = []
    for path in frame_paths:
        img = cv2.imread(path)
        shapes.append(img.shape[0])
        shapes.append(img.shape[1])
        # max_shape = max(max_shape, img.shape[0])
        # max_shape = max(max_shape, img.shape[1])

    print("Video", video_idx, "average shape:", sum(shapes)/len(shapes))
    if sum(shapes) / len(shapes) < 50:
        final_prediction = 1
        final_prediction_wt = 1
        logger.info(f"Video: {video_idx}, soccer ball shortcut prediction as 1")
    else:
        det_scores_kept = []
        for idx, pred in enumerate(result['predictions']):
            if len(pred['scores']) > 0:
                # print(pred['scores'][0])
                if pred['scores'][0] > args.det_threshold:
                    bounding_box = pred['polygons']
                    
                    img = cv2.imread(frame_paths[idx])
                    bounding_box = [int(i) for i in bounding_box[0]]
                    cropped_image = img[bounding_box[3]:bounding_box[1], bounding_box[0]:bounding_box[4]]
                    if cropped_image.shape[0] > 10 and cropped_image.shape[1] > 10:
                        cropped_imgs.append(cropped_image)
                        det_scores_kept.append(pred['scores'][0])

        for i, rec_infer in enumerate(recognizers):
            print(recognizer_names[i])
            rec_result = rec_infer(cropped_imgs, out_dir=args.output_dir, save_vis=False, return_vis=False)
            rec_result = rec_result['predictions']

            for i, pred_rec in enumerate(rec_result):
                text = pred_rec['text']
                rec_score = pred_rec['scores']
                # print("det score", det_scores_kept[i], "rec score", rec_score, text)
                if rec_score > args.rec_threshold and text.isnumeric():
                    if len(str(text)) > 2:
                        predictions.append(int(str(text)[-2:]))
                    else:
                        predictions.append(int(text))
                
        predictions = np.array(predictions)
        
        final_prediction = scipy.stats.mode(predictions, axis=None, keepdims=False)[0]
        if np.isnan(final_prediction):
            final_prediction = -1

        if len(predictions) < 5 * len(recognizers):
            final_prediction_wt = -1
        else:
            final_prediction_wt = get_weighted_most_frequent_number(predictions)

    correct.append(final_prediction == gt)
    weighted_correct.append(final_prediction_wt == gt)

    # use the weighted prediction for submission
    output_json[str(video_idx)] = int(final_prediction_wt)

    print(f"Video: {video_idx}, Pred: {final_prediction, final_prediction_wt}, GT: {gt} Correct?: {final_prediction == gt, final_prediction_wt == gt}, {predictions}")
    logger.info(f"Video: {video_idx}, Pred: {final_prediction, final_prediction_wt}, GT: {gt} Correct?: {final_prediction == gt, final_prediction_wt == gt}, {predictions}")
    # print(f"Video: {video_idx}, Pred: {final_prediction, final_prediction_wt}, GT: {gt} Correct?: {final_prediction == gt, final_prediction_wt == gt}, {predictions}")
    # shutil.rmtree(cropped_path)
    if video_idx % 50 == 0:
        print(f"Video{video_idx} ACC: {sum(correct)}/{len(correct)}, {sum(weighted_correct)}/{len(weighted_correct)}")
        logger.info(f"Video{video_idx} ACC: {sum(correct)}/{len(correct)}, {sum(weighted_correct)}/{len(weighted_correct)}")

        with open(checkpoint_path, 'w') as f:
            json.dump(output_json, f)
    
# logger.info(f"Final Accuracy: {count(correct)}/{len(correct)} = {count(correct) / len(correct)}")
logger.info(f"Final Accuracy: {sum(correct)}/{len(correct)}")
logger.info(f"Final Accuracy weighted: {sum(weighted_correct)}/{len(weighted_correct)}")

print("Final Accuracy: ", sum(correct)/len(correct))
print("Final Accuracy weighted: ", sum(weighted_correct)/len(weighted_correct))
