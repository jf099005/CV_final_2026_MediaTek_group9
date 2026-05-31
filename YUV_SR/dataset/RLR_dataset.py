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
            num_frames = int(row[3])

            videos.append({
                "lr_yuv_path": lr_yuv_path,
                "hr_yuv_path": hr_yuv_path,
                "even_frame_path": even_frame_path,
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

        prv_hr_y, prv_hr_u, prv_hr_v = read_yuv420_10bit_frame(
            video["even_frame_path"],
            self.hr_width,
            self.hr_height,
            frame_idx,
        )

        nxt_hr_y, nxt_hr_u, nxt_hr_v = read_yuv420_10bit_frame(
            video["even_frame_path"],
            self.hr_width,
            self.hr_height,
            frame_idx + 1,
        )

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

        prv_hr_patch = prv_hr_y[
            hr_y_pos:hr_y_pos + self.hr_patch_size,
            hr_x:hr_x + self.hr_patch_size
        ]

        nxt_hr_patch = nxt_hr_y[
            hr_y_pos:hr_y_pos + self.hr_patch_size,
            hr_x:hr_x + self.hr_patch_size
        ]


        # prv_hr_patch_u = prv_hr_u[
        #     hr_y_pos // 2:(hr_y_pos + self.hr_patch_size) // 2,
        #     hr_x // 2:(hr_x + self.hr_patch_size) // 2
        # ]

        # prv_hr_patch_v = prv_hr_v[
        #     hr_y_pos // 2:(hr_y_pos + self.hr_patch_size) // 2,
        #     hr_x // 2:(hr_x + self.hr_patch_size) // 2
        # ]

        # nxt_hr_patch_u = nxt_hr_u[
        #     hr_y_pos // 2:(hr_y_pos + self.hr_patch_size) // 2,
        #     hr_x // 2:(hr_x + self.hr_patch_size) // 2
        # ]

        # nxt_hr_patch_v = nxt_hr_v[
        #     hr_y_pos // 2:(hr_y_pos + self.hr_patch_size) // 2,
        #     hr_x // 2:(hr_x + self.hr_patch_size) // 2
        # ]
        
        # prv_warped_hr_patch, _, nxt_warped_hr_patch, _ = warping_hr(
        #     raft_estimator=self.raft_estimator,
        #     b_y=lr_patch,
        #     b_u=lr_u,
        #     b_v=lr_v,
        #     e_prev_y=prv_hr_patch,
        #     e_prev_u=prv_hr_patch_u,
        #     e_prev_v=prv_hr_patch_v,
        #     e_next_y=nxt_hr_patch,
        #     e_next_u=nxt_hr_patch_u,
        #     e_next_v=nxt_hr_patch_v,
        # )


        lr_patch = lr_patch.astype("float32") / self.max_value
        hr_patch = hr_patch.astype("float32") / self.max_value
        prv_hr_patch = prv_hr_patch.astype("float32") / self.max_value
        nxt_hr_patch = nxt_hr_patch.astype("float32") / self.max_value

        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        prv_hr_patch = torch.from_numpy(prv_hr_patch).unsqueeze(0)
        nxt_hr_patch = torch.from_numpy(nxt_hr_patch).unsqueeze(0)

        return (lr_patch, prv_hr_patch, nxt_hr_patch), hr_patch