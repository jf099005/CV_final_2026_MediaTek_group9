import random
import torch
from torch.utils.data import Dataset, ConcatDataset

from utils.yuv_io import read_yuv420_10bit_frame

class YOnlySRDataset(Dataset):
    def __init__(
        self,
        lr_yuv_path,
        hr_yuv_path,
        lr_width,
        lr_height,
        hr_width,
        hr_height,
        num_frames,
        scale=2,
        lr_patch_size=96,
        samples_per_epoch=10000,
        bit_depth=10,
    ):
        self.lr_yuv_path = lr_yuv_path
        self.hr_yuv_path = hr_yuv_path

        self.lr_width = lr_width
        self.lr_height = lr_height
        self.hr_width = hr_width
        self.hr_height = hr_height

        self.num_frames = num_frames
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale

        self.samples_per_epoch = samples_per_epoch
        self.max_value = (1 << bit_depth) - 1

        assert self.hr_width == self.lr_width * scale
        assert self.hr_height == self.lr_height * scale

    def __len__(self):
        # return self.num_frames-2
        return self.samples_per_epoch

    def __getitem__(self, idx):
        idx = idx + 1
        # frame_idx = random.randint(0, self.num_frames - 1)
        frame_idx = idx#random.randint(0, self.num_frames - 1)
        prv_frame_idx = frame_idx - 1
        nxt_frame_idx = frame_idx + 1

        lr_y, _, _ = read_yuv420_10bit_frame(
            self.lr_yuv_path,
            self.lr_width,
            self.lr_height,
            frame_idx,
        )

        hr_y, _, _ = read_yuv420_10bit_frame(
            self.hr_yuv_path,
            self.hr_width,
            self.hr_height,
            frame_idx,
        )

        prv_hr_y, _, _ = read_yuv420_10bit_frame(
            self.hr_yuv_path,
            self.hr_width,
            self.hr_height,
            prv_frame_idx,
        )

        nxt_hr_y, _, _ = read_yuv420_10bit_frame(
            self.hr_yuv_path,
            self.hr_width,
            self.hr_height,
            nxt_frame_idx,
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

        lr_patch = lr_patch.astype("float32") / self.max_value
        hr_patch = hr_patch.astype("float32") / self.max_value
        prv_hr_patch = prv_hr_patch.astype("float32") / self.max_value
        nxt_hr_patch = nxt_hr_patch.astype("float32") / self.max_value

        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        prv_hr_patch = torch.from_numpy(prv_hr_patch).unsqueeze(0)
        nxt_hr_patch = torch.from_numpy(nxt_hr_patch).unsqueeze(0)

        return (lr_patch, prv_hr_patch, nxt_hr_patch), hr_patch


def build_merged_dataset(
    path_pairs,
    lr_width,
    lr_height,
    hr_width,
    hr_height,
    num_frames,
    scale=2,
    lr_patch_size=96,
    samples_per_epoch=10000,
    bit_depth=10,
):
    """Build a single merged dataset from a list of (lr_yuv_path, hr_yuv_path) pairs."""
    datasets = [
        YOnlySRDataset(
            lr_yuv_path=lr_path,
            hr_yuv_path=hr_path,
            lr_width=lr_width,
            lr_height=lr_height,
            hr_width=hr_width,
            hr_height=hr_height,
            num_frames=num_frames,
            scale=scale,
            lr_patch_size=lr_patch_size,
            samples_per_epoch=samples_per_epoch,
            bit_depth=bit_depth,
        )
        for lr_path, hr_path in path_pairs
    ]
    return ConcatDataset(datasets)