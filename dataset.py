import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

# the index gives the video--e.g. idx 1 = video 1 (num_frames x H x W x 3)
class soccernet_dataset(Dataset):
    def __init__(self, root_dir, mode):
        self.vid_dir = os.path.join(root_dir, mode, "images")           # data/train/images
        vid_names = [f for f in os.listdir(self.vid_dir) if not f.startswith(".")]
        self.vid_names = sorted(vid_names, key=lambda x: int(x))        # [0, 1, 2, ...]

        gt_file = os.path.join(root_dir, mode, f"{mode}_gt.json")
        with open(gt_file) as f:
            self.gt = json.load(f)

        assert len(self.vid_names) == len(self.gt)

    def __len__(self):
        return len(self.vid_names)

    # each video is a list of frames, each frame is shape (3, H, W)
    def __getitem__(self, idx):
        vid_name = self.vid_names[idx]
        vid_dir = os.path.join(self.vid_dir, vid_name)

        frame_names = [f for f in os.listdir(vid_dir) if not f.startswith(".")]
        frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0].split('_')[1]))
        frame_paths = [os.path.join(vid_dir, frame_name) for frame_name in frame_names]
        # print(frame_names)
        frames = []
        for frame_path in frame_paths:
            frame = read_image(frame_path)
            frames.append(frame)
        return frames, frame_paths, self.gt[vid_name]
