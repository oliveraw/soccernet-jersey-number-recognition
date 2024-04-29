import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# the index gives the video--e.g. idx 1 = video 1 (num_frames x H x W x 3)
class soccernet_dataset(Dataset):
    def __init__(self, root_dir, mode):
        self.vid_dir = os.path.join(root_dir, mode, "images")           # data/train/images
        vid_names = [f for f in os.listdir(self.vid_dir) if not f.startswith(".") and not f.endswith(".json")]
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
        # frames = []
        # for frame_path in frame_paths:
            # frame = read_image(frame_path)
            # frames.append(frame)
        return frame_paths, self.gt[vid_name]
<<<<<<< HEAD

    def get_dataset_dir(self):
        return self.vid_dir
=======
>>>>>>> b11a05adc516705c385bfdfd9080020049f79c4a


# images will not be in order
# get item will just return a single frame rather than a whole video
class soccernet_dataset_flat(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        # if you pre-generated all the file names (much faster)
        frames_file = os.path.join(root_dir, mode, "frames.json")
        with open(frames_file) as f:
            frames_dict = json.load(f)

        gt_file = os.path.join(root_dir, mode, f"{mode}_gt.json")
        with open(gt_file) as f:
            gt_dict = json.load(f)
        gt_vals = gt_dict.values()
        # print("ground truth possible values", mode, sorted(gt_vals, key=lambda x: int(x)))
        
        self.frame_paths = []
        self.gt = []
        for vid_name, frame_list in frames_dict.items():
            frame_paths = [os.path.join(root_dir, mode, "images", vid_name, f) for f in frame_list]
            self.frame_paths.extend(frame_paths)
            self.gt.extend([gt_dict[vid_name]] * len(frame_paths))

        # otherwise, need to get all the files
        # data_dir = os.path.join(root_dir, mode, "images")           # data/train/images
        # vid_names = [f for f in os.listdir(data_dir) if not f.startswith(".")]
        # vid_names = sorted(vid_names, key=lambda x: int(x))        # [0, 1, 2, ...]
        # self.frame_paths = []
        # self.gt = []
        # for vid_name in vid_names:
        #     vid_dir = os.path.join(data_dir, vid_name)
        #     frame_names = [f for f in os.listdir(vid_dir) if not f.startswith(".")]
        #     frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0].split('_')[1]))
        #     frame_paths_for_this_vid = [os.path.join(vid_dir, f) for f in frame_names]
        #     self.frame_paths.extend(frame_paths_for_this_vid)
        #     self.gt.extend([gt_dict[vid_name]] * len(frame_paths_for_this_vid))

        assert len(self.frame_paths) == len(self.gt)

        self.transform = transform
    
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame = Image.open(self.frame_paths[idx])

        if self.transform:
            frame = self.transform(frame)
        
        return frame, self.gt[idx], self.frame_paths[idx]


# generate_all_file_names('./data/train/images', './data/train/frames.txt')
# generate_all_file_names('./data/test/images', './data/test/frames.txt')
# generate_all_file_names('./data/challenge/images', './data/challenge/frames.txt')
def generate_all_file_names(data_dir, output_file):
    frame_paths_per_vid = {}
    vid_names = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    vid_names = sorted(vid_names, key=lambda x: int(x))
    for vid_name in vid_names:
        print(vid_name)
        vid_dir = os.path.join(data_dir, vid_name)
        frame_names = [f for f in os.listdir(vid_dir) if not f.startswith(".")]
        frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0].split('_')[1]))
        frame_paths_per_vid[vid_name] = frame_names
    
    with open(output_file, 'w') as f:
        json.dump(frame_paths_per_vid, f)