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
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.scale = scale
        self.max_value = (1 << bit_depth) - 1

        # self.model = EDSRSmall(
        #     in_channels=1,
        #     out_channels=1,
        #     num_features=64,
        #     num_blocks=8,
        #     scale=scale,
        # ).to(self.device)

        self.model = EDSRSmall(
            in_channels=1,
            out_channels=1,
            num_features=48,
            num_blocks=8,
            scale=scale,
        ).to(device)


        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )

        self.model.eval()

    @torch.no_grad()
    def upscale_y(self, y):
        """
        Input:
            y: np.ndarray, shape = (H, W), uint16, 10-bit Y channel

        Output:
            sr_y: np.ndarray, shape = (H*scale, W*scale), uint16
        """
        y_float = y.astype(np.float32) / self.max_value

        y_tensor = torch.from_numpy(y_float)
        y_tensor = y_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        sr_y = self.model(y_tensor)

        sr_y = sr_y.squeeze(0).squeeze(0).cpu().numpy()
        sr_y = np.clip(sr_y * self.max_value, 0, self.max_value)
        sr_y = np.round(sr_y).astype(np.uint16)

        return sr_y

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
        sr_y = self.upscale_y(y)

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