import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = '/mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9'
sys.path.append(PROJECT_ROOT)

from utils.RAFTFlow import RAFTFlowEstimator
from utils.GenerateFrame import backward_warp_plane_with_mask

_BIT_SCALE = 1023.0


class PNGSRDataset(Dataset):
    def __init__(
        self,
        hr_png_dir,
        scale=2,
        lr_patch_size=96,
        samples_per_epoch=1000,
        train_ratio=0.8,
        dataset_type="train",
    ):
        self.hr_png_dir = hr_png_dir
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.samples_per_epoch = samples_per_epoch
        self.train_ratio = train_ratio
        self.dataset_type = dataset_type

        self.image_paths = sorted(
            os.path.join(hr_png_dir, f)
            for f in os.listdir(hr_png_dir)
            if f.lower().endswith(".png")
        )
        assert len(self.image_paths) > 0, f"No PNG files found in {hr_png_dir}"

        self.raft_estimator = RAFTFlowEstimator()

    def __len__(self):
        return self.samples_per_epoch

    def _rgb_to_y(self, img_rgb: np.ndarray) -> np.ndarray:
        """BT.709 luma, returns float32 in [0, 1]."""
        r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y.astype(np.float32)

    def _load_hr_y(self, path) -> np.ndarray:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return self._rgb_to_y(img)

    def _load_frame(self, path):
        """Load HR image once; derive LR via bicubic downsampling. Returns hr_y, lr_y, lr_h, lr_w."""
        hr_img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = hr_h // self.scale, hr_w // self.scale
        lr_pil = Image.fromarray((hr_img * 255).astype(np.uint8)).resize(
            (lr_w, lr_h), Image.BICUBIC
        )
        lr_img = np.array(lr_pil, dtype=np.float32) / 255.0
        return self._rgb_to_y(hr_img), self._rgb_to_y(lr_img), lr_h, lr_w

    def __getitem__(self, _idx):
        num_frames = len(self.image_paths)
        split_idx = int(num_frames * self.train_ratio)

        if self.dataset_type == "train":
            frame_idx = random.randint(1, split_idx - 2)
        else:
            frame_idx = random.randint(split_idx + 1, num_frames - 2)

        prv_frame_idx = frame_idx - 1
        nxt_frame_idx = frame_idx + 1

        # Load current frame: derive both HR and LR from the same file open
        hr_y, lr_y, lr_h, lr_w = self._load_frame(self.image_paths[frame_idx])
        prv_hr_y = self._load_hr_y(self.image_paths[prv_frame_idx])
        nxt_hr_y = self._load_hr_y(self.image_paths[nxt_frame_idx])

        x = random.randint(0, lr_w - self.lr_patch_size)
        y = random.randint(0, lr_h - self.lr_patch_size)

        lr_patch = lr_y[y : y + self.lr_patch_size, x : x + self.lr_patch_size].copy()

        hr_x_pos = x * self.scale
        hr_y_pos = y * self.scale

        hr_patch = hr_y[
            hr_y_pos : hr_y_pos + self.hr_patch_size,
            hr_x_pos : hr_x_pos + self.hr_patch_size,
        ].copy()
        prv_hr_patch = prv_hr_y[
            hr_y_pos : hr_y_pos + self.hr_patch_size,
            hr_x_pos : hr_x_pos + self.hr_patch_size,
        ].copy()
        nxt_hr_patch = nxt_hr_y[
            hr_y_pos : hr_y_pos + self.hr_patch_size,
            hr_x_pos : hr_x_pos + self.hr_patch_size,
        ].copy()

        # Scale to 10-bit range for RAFT compatibility
        # lr_patch_10 = (lr_patch * _BIT_SCALE).astype(np.float32)
        # prv_hr_patch_10 = (prv_hr_patch * _BIT_SCALE).astype(np.float32)
        # nxt_hr_patch_10 = (nxt_hr_patch * _BIT_SCALE).astype(np.float32)

        # Upscale LR patch to HR resolution for flow estimation
        # Use INTER_CUBIC (not INTER_AREA) — INTER_AREA degrades to nearest-neighbor when upscaling
        # b_raft_y = cv2.resize(
        #     lr_patch_10,
        #     (self.hr_patch_size, self.hr_patch_size),
        #     interpolation=cv2.INTER_CUBIC,
        # )

        # flow_curr2prev = self.raft_estimator.compute_flow(b_raft_y, prv_hr_patch_10)
        # flow_curr2next = self.raft_estimator.compute_flow(b_raft_y, nxt_hr_patch_10)

        # prv_warped_hr_patch, _ = backward_warp_plane_with_mask(prv_hr_patch_10, flow_curr2prev)
        # nxt_warped_hr_patch, _ = backward_warp_plane_with_mask(nxt_hr_patch_10, flow_curr2next)

        lr_patch = lr_patch.astype(np.float32)
        hr_patch = hr_patch.astype(np.float32)
        # prv_warped_hr_patch = (prv_warped_hr_patch / _BIT_SCALE).astype(np.float32)
        # nxt_warped_hr_patch = (nxt_warped_hr_patch / _BIT_SCALE).astype(np.float32)

        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)

        prv_hr_patch = torch.from_numpy(prv_hr_patch).unsqueeze(0)
        nxt_hr_patch = torch.from_numpy(nxt_hr_patch).unsqueeze(0)

        # prv_warped_hr_patch = torch.from_numpy(prv_warped_hr_patch).unsqueeze(0)
        # nxt_warped_hr_patch = torch.from_numpy(nxt_warped_hr_patch).unsqueeze(0)

        # return (lr_patch, prv_warped_hr_patch, nxt_warped_hr_patch), hr_patch
        return (lr_patch, prv_hr_patch, nxt_hr_patch), hr_patch
