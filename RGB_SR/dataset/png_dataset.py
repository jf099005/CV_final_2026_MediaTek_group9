import csv
import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = '/mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9'
sys.path.append(PROJECT_ROOT)


def _sorted_pngs(directory):
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    )


def load_video_list(list_path):
    entries = []
    with open(list_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            lr_png_dir      = row[0].strip()
            even_hr_png_dir = row[1].strip()
            hr_png_dir      = row[2].strip()
            num_frames      = int(row[3].strip())
            entries.append({
                "lr_frames":   _sorted_pngs(lr_png_dir),
                "even_frames": _sorted_pngs(even_hr_png_dir),
                "hr_frames":   _sorted_pngs(hr_png_dir),
                "num_frames":  num_frames,
            })
    assert len(entries) > 0, f"No entries in {list_path}"
    return entries


class PNGSRDataset(Dataset):
    """
    RGB SR dataset using real encoded PNG frames from the bitstream pipeline.

    CSV columns: lr_png_dir, even_hr_png_dir, hr_png_dir, num_frames

    Temporal layout (interleaved odd/even):
        ... even[k-1]  odd[k]  even[k] ...

    For odd frame k:
        lr     = lr_frames[k]       (compressed LR, 1920x1080)
        hr     = hr_frames[k]       (original HR ground truth, 3840x2160)
        prv_hr = even_frames[k-1]   (HR even frame before odd k)
        nxt_hr = even_frames[k]     (HR even frame after odd k)

    Valid range: k in [1, num_frames-2] ensures both prv and nxt exist.
    """

    def __init__(
        self,
        list_path,
        scale=2,
        lr_patch_size=96,
        samples_per_epoch=1000,
        train_ratio=0.8,
        dataset_type="train",
    ):
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.samples_per_epoch = samples_per_epoch

        videos = load_video_list(list_path)

        self.valid_samples = []  # (video_entry, odd_frame_idx)
        for v in videos:
            n = v["num_frames"]
            split_idx = int(n * train_ratio)

            if dataset_type == "train":
                valid_range = range(1, split_idx - 1)
            else:
                valid_range = range(split_idx + 1, n - 1)

            self.valid_samples.extend((v, k) for k in valid_range)

        assert len(self.valid_samples) > 0, (
            f"No valid frames for dataset_type='{dataset_type}' in {list_path}"
        )

    def __len__(self):
        return self.samples_per_epoch

    def _load_bgr(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None, f"Failed to load: {path}"
        return img.astype(np.float32) / 255.0

    def __getitem__(self, _idx):
        video, k = random.choice(self.valid_samples)

        lr_bgr     = self._load_bgr(video["lr_frames"][k])
        hr_bgr     = self._load_bgr(video["hr_frames"][k])
        prv_hr_bgr = self._load_bgr(video["even_frames"][k - 1])
        nxt_hr_bgr = self._load_bgr(video["even_frames"][k])

        lr_h, lr_w = lr_bgr.shape[:2]

        x = random.randint(0, lr_w - self.lr_patch_size)
        y = random.randint(0, lr_h - self.lr_patch_size)
        hr_x, hr_y = x * self.scale, y * self.scale

        lr_patch     = lr_bgr[y    : y    + self.lr_patch_size,  x    : x    + self.lr_patch_size].copy()
        hr_patch     = hr_bgr[hr_y : hr_y + self.hr_patch_size,  hr_x : hr_x + self.hr_patch_size].copy()
        prv_hr_patch = prv_hr_bgr[hr_y : hr_y + self.hr_patch_size, hr_x : hr_x + self.hr_patch_size].copy()
        nxt_hr_patch = nxt_hr_bgr[hr_y : hr_y + self.hr_patch_size, hr_x : hr_x + self.hr_patch_size].copy()

        # HWC → CHW
        lr_patch     = torch.from_numpy(lr_patch.transpose(2, 0, 1))
        hr_patch     = torch.from_numpy(hr_patch.transpose(2, 0, 1))
        prv_hr_patch = torch.from_numpy(prv_hr_patch.transpose(2, 0, 1))
        nxt_hr_patch = torch.from_numpy(nxt_hr_patch.transpose(2, 0, 1))

        return (lr_patch, prv_hr_patch, nxt_hr_patch), hr_patch
