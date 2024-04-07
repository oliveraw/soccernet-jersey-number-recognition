
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
from tqdm import tqdm
from collections import defaultdict
from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat, generate_all_file_names


#%%

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.6, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
parser.add_argument('--det_config_path', default='mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py', type=str, help='python file which defines architecture and training configurations')
parser.add_argument('--det_weights_path', default='mmocr/soccernet-dbnetpp/epoch_10.pth', type=str, help='weights for the finetuned detector')
parser.add_argument('--rec_weights_path', default='mmocr/soccernet-dbnetpp/svtr-small/epoch_8.pth', type=str, help='weights for the finetuned recognitor')
parser.add_argument('--rec_config_path', default='mmocr/configs/textrecog/svtr/svtr-small_5e_soccernet.py', type=str, help='python file which defines architecture and training configurations')


args = parser.parse_args()

# toggle between INFO, DEBUG
logfile = f"logs/soccernet-{os.getenv('SLURM_JOB_ID')}-info.log"
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

# train_dataset = soccernet_dataset(args.data_path, "train")
test_dataset = soccernet_dataset(args.data_path, "test")
# logger.info(f"Num videos in train dataset: {len(train_dataset)}")
logger.info(f"Num videos in test dataset: {len(test_dataset)}")
# debug_dataset = soccernet_dataset(args.data_path, "debug")
# logger.info(f"Num videos in test dataset: {len(debug_dataset)}")

#%%

det_infer = TextDetInferencer(model=args.det_config_path, weights=args.det_weights_path, device = device)
rec_infer = TextRecInferencer(model=args.rec_config_path, weights=args.rec_weights_path, device = device)

correct = []
cropped_path = args.data_path+'/cropped'
print(cropped_path)

for video_idx, (frame_paths, gt) in enumerate(test_dataset):

    predictions = []
    result = det_infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=False)
    # idx_to_rec = []
    cropped_imgs = []

    # create folder for cropped imgs
    os.mkdir(cropped_path)

    for idx, pred in enumerate(result['predictions']):
        
        if len(pred['scores']) > 0:
            
            if pred['scores'][0] > args.det_threshold:
                bounding_box = pred['polygons']
                # idx_to_rec.append(idx)
                
                # save cropped imgs
                img = cv2.imread(frame_paths[idx])
                bounding_box = [int(i) for i in bounding_box[0]]
                cropped_image = img[bounding_box[3]:bounding_box[1], bounding_box[0]:bounding_box[4]]
                img_path = f'{cropped_path}/{idx}.png'
                try: # there's some error that I've decided to figure out later
                    cv2.imwrite(img_path, cropped_image)
                    cropped_imgs.append(img_path)
                except:
                    pass
    
    # frame_paths_rec = [frame_paths[i] for i in idx_to_rec]
    
    rec_result = rec_infer(cropped_imgs, out_dir=args.output_dir, save_vis=False, return_vis=False)
    rec_result = rec_result['predictions']

    for i, pred_rec in enumerate(rec_result):
        text = pred_rec['text']
        if text.isnumeric():
            if len(str(text)) > 2:
                predictions.append(int(str(text)[-2:]))
            else:
                predictions.append(int(text))
            
    predictions = np.array(predictions)
    if np.isnan(final_prediction):
        final_prediction = -1
    final_prediction = scipy.stats.mode(predictions, axis=None, keepdims=False)[0]

    logger.info(f"Video: {video_idx}, Prediction: {final_prediction}, Ground truth: {gt} Correct?: {final_prediction == gt}, {predictions}")
    correct.append(final_prediction == gt)
    
    # remove the folder
    shutil.rmtree(cropped_path)


logger.info(f"Final Accuracy: {sum(correct)}/{len(correct)}")
print("Final Accuracy: ", sum(correct)/len(correct))
