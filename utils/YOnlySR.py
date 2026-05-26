import cv2
import numpy as np
import torch

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUV_SR_DIR = os.path.join(PROJECT_ROOT, "YUV_SR")

sys.path.append(YUV_SR_DIR)

from models.edsr import EDSREnd2EndModel

class YOnlySR:
    def __init__(
        self,
        model_path,
        scale=2,
        bit_depth=10,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.scale = scale
        self.max_value = (1 << bit_depth) - 1

        self.model = EDSREnd2EndModel(
            in_channels=1,
            out_channels=1,
            num_features=64,
            num_blocks=8,
            scale=scale,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )

        self.model.eval()

    def _to_tensor(self, arr):
        return (
            torch.from_numpy(arr.astype(np.float32) / self.max_value)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

    @torch.no_grad()
    def upscale_y(self, y, prv_y_hr, nxt_y_hr):
        """
        y:        np.ndarray (H, W) uint16, 10-bit LR Y channel
        prv_y_hr: np.ndarray (H*scale, W*scale) uint16, HR Y of previous frame
        nxt_y_hr: np.ndarray (H*scale, W*scale) uint16, HR Y of next frame

        Returns: sr_y np.ndarray (H*scale, W*scale) uint16
        """
        y_tensor   = self._to_tensor(y)
        prv_tensor = self._to_tensor(prv_y_hr)
        nxt_tensor = self._to_tensor(nxt_y_hr)

        sr_y = self.model(y_tensor, prv_tensor, nxt_tensor)

        sr_y = sr_y.squeeze(0).squeeze(0).cpu().numpy()
        sr_y = np.clip(sr_y * self.max_value, 0, self.max_value)
        sr_y = np.round(sr_y).astype(np.uint16)

        return sr_y

    def upscale_yuv420(self, y, u, v, prv_y_hr, nxt_y_hr):
        """
        y, u, v:          LR planes (H, W), (H/2, W/2), (H/2, W/2)
        prv_y_hr, nxt_y_hr: HR Y planes for previous/next frames

        Returns: sr_y, sr_u, sr_v at 2x resolution
        """
        sr_y = self.upscale_y(y, prv_y_hr, nxt_y_hr)

        out_h, out_w = sr_y.shape

        sr_u = cv2.resize(u, (out_w // 2, out_h // 2), interpolation=cv2.INTER_CUBIC)
        sr_v = cv2.resize(v, (out_w // 2, out_h // 2), interpolation=cv2.INTER_CUBIC)

        sr_u = np.clip(sr_u, 0, self.max_value).round().astype(np.uint16)
        sr_v = np.clip(sr_v, 0, self.max_value).round().astype(np.uint16)

        return sr_y, sr_u, sr_v
