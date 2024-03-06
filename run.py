import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer
from torchvision.io import ImageReadMode

from dataset import soccernet_dataset

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--det_threshold', default=0.3, type=float)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset, the dir with (train, test, challenge) directories')
parser.add_argument('--output_dir', default='./outputs', type=str, help='directory to store outputs')
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

infer = MMOCRInferencer(det='dbnetpp', device=device)

for (frames, frame_paths, gt) in train_dataset:
    # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output
    result = infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)
    for idx, pred in enumerate(result['predictions']):
        det_scores = pred['det_scores']
        over_threshold = len(det_scores) != 0 and any(i >= args.det_threshold for i in det_scores)
        print(det_scores)
        if over_threshold:
            filename = frame_paths[idx].split('/')[-1]
            plt.figure()
            plt.title(det_scores)
            plt.imshow(result['visualization'][0])
            plt.savefig(f"{args.output_dir}/{filename}")
            logger.debug(f"Saving figure {filename}")

    # for path in frame_paths:
    #     # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output
    #     result = infer(path, out_dir=args.output_dir, save_vis=False, return_vis=True)
    #     det_scores = result['predictions'][0]['det_scores']
    #     over_threshold = len(det_scores) != 0 and all(i >= args.det_threshold for i in det_scores)
    #     print(result['predictions'][0]['det_scores'])
    #     if over_threshold: 
    #         plt.figure()
    #         plt.imshow(result['visualization'][0])
    #         plt.savefig(path)
    #         logging.debug(f"Saving figure {path}")