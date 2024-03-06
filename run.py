import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mmocr.apis import MMOCRInferencer
from torchvision.io import ImageReadMode
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
from torch.utils.data import DataLoader

from soccernet_dataset import soccernet_dataset, soccernet_dataset_flat, generate_all_file_names
from resnet import ResNet18, ResNet50

parser = argparse.ArgumentParser(description='EECS 545 SoccerNet Jersey Number Recognition')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--num_epochs_per_eval', default=100, type=int)
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

# model
transforms = transforms.Compose([
    transforms.Resize((128, 64)),
    transforms.Normalize((127.4395, 135.9471,  84.0932), (38.5333, 38.7357, 47.5903))
])

model = ResNet18(num_classes=101).to(device)
summary(model, (3, 128, 64))

# data
train_dataset = soccernet_dataset_flat(args.data_path, "train", transforms)
test_dataset = soccernet_dataset_flat(args.data_path, "test", transforms)
logger.info(f"Num frames in train dataset: {len(train_dataset)}")
logger.info(f"Num frames in test dataset: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# training 
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=args.lr)
for epoch in range(args.num_epochs+1):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        running_loss += loss.item()
    logger.info(f'Epoch {epoch} loss: {running_loss}')

    if epoch % args.num_epochs_per_eval == 0:
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(test_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            logger.info(f'Epoch {epoch} test loss: {running_loss}')
        model.train()

# for i in range(10):
#     frame, gt = train_dataset[i]
#     frame = frame.permute(1, 2, 0)
#     plt.figure()
#     plt.imshow(frame)
#     plt.savefig(f'outputs/{i}.jpg')




##########################non-flat###############################
# train_dataset = soccernet_dataset(args.data_path, "train")
# test_dataset = soccernet_dataset(args.data_path, "test")
# logger.info(f"Num videos in train dataset: {len(train_dataset)}")
# logger.info(f"Num videos in test dataset: {len(test_dataset)}")

# infer = MMOCRInferencer(rec='dbnetpp', device=device)

# for (frames, frame_paths, gt) in train_dataset:
#     # output of inferencer is in this format: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html#output
#     result = infer(frame_paths, out_dir=args.output_dir, save_vis=False, return_vis=True)
#     for idx, pred in enumerate(result['predictions']):
#         det_scores = pred['det_scores']
#         over_threshold = len(det_scores) != 0 and any(i >= args.det_threshold for i in det_scores)
#         print(det_scores, rec_texts)
#         if over_threshold:
#             filename = frame_paths[idx].split('/')[-1]
#             plt.figure()
#             plt.title(det_scores)
#             plt.imshow(result['visualization'][0])
#             plt.savefig(f"{args.output_dir}/{filename}")
#             logger.debug(f"Saving figure {filename}")
