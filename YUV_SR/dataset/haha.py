import os
import random
import csv
import sys
import torch
from torch.utils.data import Dataset
from optical_flow_warping import warping_hr

PROJECT_ROOT = '/mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9'
# utils_DIR = os.path.join(PROJECT_ROOT, "YUV_SR")
sys.path.append(PROJECT_ROOT)

from utils.yuv_io import read_yuv420_10bit_frame
from utils.RAFTFlow import RAFTFlowEstimator

def load_video_list(list_path):
    videos = []

    with open(list_path, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 0:
                continue

            # 跳過註解
            if row[0].startswith("#"):
                continue

            lr_yuv_path = row[0]
            hr_yuv_path = row[2]
            even_frame_path = row[1]
            warped_frame_path = row[3]
            num_frames = int(row[4])

            videos.append({
                "lr_yuv_path": lr_yuv_path,
                "hr_yuv_path": hr_yuv_path,
                "warped_hr_yuv_path": even_frame_path,
                "num_frames": num_frames,
            })

    return videos


class YOnlySRDataset(Dataset):
    """
    Training dataset:
    random video + random frame + random patch
    """

    def __init__(
        self,
        list_path,
        # even_frame_path,
        lr_width,
        lr_height,
        hr_width,
        hr_height,
        scale=2,
        lr_patch_size=96,
        samples_per_epoch=1000,
        bit_depth=10,
        train_ratio=0.8,
        dataset_type="train",
    ):
        self.videos = load_video_list(list_path)
        self.dataset_type = dataset_type
        # self.even_frame_path = even_frame_path

        self.lr_width = lr_width
        self.lr_height = lr_height
        self.hr_width = hr_width
        self.hr_height = hr_height

        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale

        self.samples_per_epoch = samples_per_epoch
        self.max_value = (1 << bit_depth) - 1
        self.train_ratio = train_ratio

        self.raft_estimator = RAFTFlowEstimator()  # Initialize your RAFT estimator here

        assert self.hr_width == self.lr_width * scale
        assert self.hr_height == self.lr_height * scale
        assert len(self.videos) > 0

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        video = random.choice(self.videos)

        num_frames = video["num_frames"]
        split_idx = int(num_frames * self.train_ratio)

        if self.dataset_type == "train":
            frame_idx = random.randint(1, split_idx - 2)
        else:
            frame_idx = random.randint(split_idx + 1, num_frames - 2)

        lr_y, lr_u, lr_v = read_yuv420_10bit_frame(
            video["lr_yuv_path"],
            self.lr_width,
            self.lr_height,
            frame_idx,
        )

        hr_y, hr_u, hr_v = read_yuv420_10bit_frame(
            video["hr_yuv_path"],
            self.hr_width,
            self.hr_height,
            frame_idx,
        )

        

        # warped_hr_y, warped_hr_u, warped_hr_v = read_yuv420_10bit_frame(
        #     video["warped_hr_yuv_path"],
        #     self.hr_width,
        #     self.hr_height,
        #     frame_idx,
        # )

        x = random.randint(0, self.lr_width - self.lr_patch_size)
        y = random.randint(0, self.lr_height - self.lr_patch_size)

        lr_patch = lr_y[
            y:y + self.lr_patch_size,
            x:x + self.lr_patch_size
        ]

        hr_x = x * self.scale
        hr_y_pos = y * self.scale

        hr_patch = hr_y[
            hr_y_pos:hr_y_pos + self.hr_patch_size,
            hr_x:hr_x + self.hr_patch_size
        ]

        warped_hr_patch = warped_hr_y[
            hr_y_pos:hr_y_pos + self.hr_patch_size,
            hr_x:hr_x + self.hr_patch_size
        ]



        lr_patch = lr_patch.astype("float32") / self.max_value
        hr_patch = hr_patch.astype("float32") / self.max_value
        warped_hr_patch = warped_hr_patch.astype("float32") / self.max_value
        # nxt_hr_patch = nxt_hr_patch.astype("float32") / self.max_value

        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        warped_hr_patch = torch.from_numpy(warped_hr_patch).unsqueeze(0)
        # nxt_hr_patch = torch.from_numpy(nxt_hr_patch).unsqueeze(0)

        return (lr_patch, warped_hr_patch), hr_patch