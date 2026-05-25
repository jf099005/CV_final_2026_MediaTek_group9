import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PNGSRDataset(Dataset):
    def __init__(
        self,
        hr_png_dir,
        scale=2,
        lr_patch_size=96,
        samples_per_epoch=10000,
    ):
        self.hr_png_dir = hr_png_dir
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.samples_per_epoch = samples_per_epoch

        self.image_paths = sorted(
            os.path.join(hr_png_dir, f)
            for f in os.listdir(hr_png_dir)
            if f.lower().endswith(".png")
        )
        assert len(self.image_paths) > 0, f"No PNG files found in {hr_png_dir}"

    def __len__(self):
        return self.samples_per_epoch

    def _rgb_to_y(self, img_rgb: np.ndarray) -> np.ndarray:
        """BT.709 luma, returns float32 in [0, 1]."""
        r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y.astype(np.float32)

    def __getitem__(self, idx):
        path = random.choice(self.image_paths)
        hr_img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = hr_h // self.scale, hr_w // self.scale

        lr_pil = Image.fromarray((hr_img * 255).astype(np.uint8)).resize(
            (lr_w, lr_h), Image.BICUBIC
        )
        lr_img = np.array(lr_pil, dtype=np.float32) / 255.0

        hr_y = self._rgb_to_y(hr_img)
        lr_y = self._rgb_to_y(lr_img)

        x = random.randint(0, lr_w - self.lr_patch_size)
        y = random.randint(0, lr_h - self.lr_patch_size)

        lr_patch = lr_y[y : y + self.lr_patch_size, x : x + self.lr_patch_size]

        hr_x = x * self.scale
        hr_y_pos = y * self.scale
        hr_patch = hr_y[
            hr_y_pos : hr_y_pos + self.hr_patch_size,
            hr_x : hr_x + self.hr_patch_size,
        ]

        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)

        return lr_patch, hr_patch
