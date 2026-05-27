import cv2
import numpy as np
import torch

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUV_SR_DIR = os.path.join(PROJECT_ROOT, "YUV_SR")

sys.path.append(YUV_SR_DIR)

from models.edsr_small import EDSRSmall

class YOnlySR:
    def __init__(
        self,
        model_path,
        scale=2,
        bit_depth=10,
        in_channels=1,
        out_channels=1,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.scale = scale
        self.max_value = (1 << bit_depth) - 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = EDSRSmall(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=64,
            num_blocks=8,
            scale=scale,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )

        self.model.eval()

    @torch.no_grad()
    def _predict(self, y, u=None, v=None):
        if self.in_channels == 1:
            y_float = y.astype(np.float32) / self.max_value
            y_tensor = torch.from_numpy(y_float)
            y_tensor = y_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        elif self.in_channels == 3:
            if u is None or v is None:
                raise ValueError("Input U/V channels are required for 3-channel SR prediction.")

            if y.ndim == 2:
                x = np.stack([y, u, v], axis=0)
            elif y.ndim == 3:
                x = y
            else:
                raise ValueError("Expected Y input shape HxW or CxHxW for 3-channel prediction.")

            x_float = x.astype(np.float32) / self.max_value
            y_tensor = torch.from_numpy(x_float).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unsupported in_channels: {self.in_channels}")

        sr = self.model(y_tensor)
        sr = sr.squeeze(0).cpu().numpy()
        sr = np.clip(sr * self.max_value, 0, self.max_value)
        sr = np.round(sr).astype(np.uint16)

        return sr

    @torch.no_grad()
    def upscale_y(self, y):
        """
        Input:
            y: np.ndarray, shape = (H, W), uint16, 10-bit Y channel

        Output:
            sr_y: np.ndarray, shape = (H*scale, W*scale), uint16
        """
        if self.in_channels != 1:
            raise ValueError("upscale_y() only supports in_channels=1.")

        sr = self._predict(y)

        if self.out_channels == 1:
            return sr

        return sr[0]

    def upscale_yuv420(self, y, u, v):
        """
        Input:
            y: H x W
            u: H/2 x W/2
            v: H/2 x W/2

        Output:
            sr_y: H*2 x W*2
            sr_u: H x W
            sr_v: H x W
        """
        if self.in_channels == 1:
            sr = self._predict(y)
        else:
            sr = self._predict(y, u, v)

        if self.out_channels == 1:
            sr_y = sr
            out_h, out_w = sr_y.shape

            sr_u = cv2.resize(
                u,
                (out_w // 2, out_h // 2),
                interpolation=cv2.INTER_CUBIC,
            )

            sr_v = cv2.resize(
                v,
                (out_w // 2, out_h // 2),
                interpolation=cv2.INTER_CUBIC,
            )

            sr_u = np.clip(sr_u, 0, self.max_value).round().astype(np.uint16)
            sr_v = np.clip(sr_v, 0, self.max_value).round().astype(np.uint16)

            return sr_y, sr_u, sr_v

        sr_y = sr[0]
        sr_u_full = sr[1]
        sr_v_full = sr[2]

        out_h, out_w = sr_y.shape

        sr_u = cv2.resize(
            sr_u_full,
            (out_w // 2, out_h // 2),
            interpolation=cv2.INTER_AREA,
        )

        sr_v = cv2.resize(
            sr_v_full,
            (out_w // 2, out_h // 2),
            interpolation=cv2.INTER_AREA,
        )

        sr_u = np.clip(sr_u, 0, self.max_value).round().astype(np.uint16)
        sr_v = np.clip(sr_v, 0, self.max_value).round().astype(np.uint16)

        return sr_y, sr_u, sr_v