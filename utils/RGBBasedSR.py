import cv2
import json
import numpy as np
import torch
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RGB_SR_DIR = os.path.join(PROJECT_ROOT, "RGB_SR")
sys.path.insert(0, RGB_SR_DIR)

from models.carn_v1 import CARNEnd2EndModel

_DEFAULT_CONFIG = os.path.join(RGB_SR_DIR, "models", "model_config.json")


class RGBBasedSR:
    """
    RGB super-resolution wrapper around EDSREnd2EndModel.

    Accepts 10-bit YUV420 planes, converts to BGR [0,1] internally using
    BT.709 full-range coefficients, runs the model, and converts the output
    back to 10-bit YUV420.
    """

    def __init__(self, model_path, bit_depth=10, device=None, config_path=None):
        if config_path is None:
            config_path = _DEFAULT_CONFIG

        with open(config_path) as f:
            cfg = json.load(f)
        mc = cfg.get("model", {})

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.scale = mc.get("scale", 2)
        self.max_value = (1 << bit_depth) - 1
        self.use_half = mc.get("use_half", False) and device != "cpu"

        # Tiling: tile_h/tile_w in base-resolution coordinates; None = no tiling.
        self.tile_h = mc.get("tile_h", None)
        self.tile_w = mc.get("tile_w", None)
        self.tile_overlap = mc.get("tile_overlap", 8)

        model_type = mc.get("type", "edsr")

        if model_type == "carn":
            # Backbone pretrained weights are loaded inside from_config via
            # carn_pretrained_path.  model_path is only used when a full
            # combined (backbone + fusion) checkpoint is available.
            self.model = CARNEnd2EndModel.from_config(config_path).to(self.device)
            if model_path is not None:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                if any(k.startswith("carn_layer.") for k in state):
                    self.model.load_state_dict(state)
                    print(f"[RGBBasedSR] Loaded full CARN combined checkpoint: {model_path}")
                else:
                    print("[RGBBasedSR] model_path is backbone-only; using backbone from config.")
        else:
            raise NotImplementedError()

        if self.use_half:
            self.model = self.model.half()
        self.model.eval()

    # ------------------------------------------------------------------
    # Color space helpers
    # ------------------------------------------------------------------

    def _yuv420_to_bgr(self, y, u, v):
        """
        10-bit YUV420 → BGR float32 [0, 1] using BT.709 limited-range.
        Matches ffmpeg default: Y∈[64,940], Cb/Cr∈[64,960], neutral=512.

        y : (H, W)      uint16
        u : (H/2, W/2)  uint16
        v : (H/2, W/2)  uint16
        """
        H, W = y.shape
        scale = (self.max_value + 1) / 256.0
        y_black, y_range = 16 * scale, 219 * scale
        c_neutral, c_scale = 128 * scale, 224 * scale   # c_scale=896 for 10-bit → E'_Cb ∈ [-0.5,+0.5]

        Y = (y.astype(np.float32) - y_black) / y_range

        # Upsample chroma to luma resolution
        Cb = (cv2.resize(u.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
              - c_neutral) / c_scale
        Cr = (cv2.resize(v.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
              - c_neutral) / c_scale

        # BT.709 YCbCr → R, G, B
        R = Y + 1.5748 * Cr
        G = Y - 0.1873 * Cb - 0.4681 * Cr
        B = Y + 1.8556 * Cb

        bgr = np.stack([B, G, R], axis=-1)   # cv2 BGR order
        return np.clip(bgr, 0.0, 1.0).astype(np.float32)

    def _bgr_to_yuv420(self, bgr):
        """
        BGR float32 [0, 1] → 10-bit YUV420 using BT.709 limited-range.
        Matches ffmpeg default: Y∈[64,940], Cb/Cr∈[64,960], neutral=512.

        bgr : (H, W, 3) float32
        Returns y (H,W), u (H/2,W/2), v (H/2,W/2)  uint16
        """
        B, G, R = bgr[..., 0], bgr[..., 1], bgr[..., 2]

        # BT.709 R, G, B → YCbCr
        Y  =  0.2126 * R + 0.7152 * G + 0.0722 * B
        Cb = -0.1146 * R - 0.3854 * G + 0.5000 * B
        Cr =  0.5000 * R - 0.4542 * G - 0.0458 * B

        # Downsample chroma 2× (average 2×2 blocks)
        Cb_ds = (Cb[0::2, 0::2] + Cb[1::2, 0::2] + Cb[0::2, 1::2] + Cb[1::2, 1::2]) / 4.0
        Cr_ds = (Cr[0::2, 0::2] + Cr[1::2, 0::2] + Cr[0::2, 1::2] + Cr[1::2, 1::2]) / 4.0

        # [0,1] → limited-range
        scale = (self.max_value + 1) / 256.0
        y_black, y_range = 16 * scale, 219 * scale
        c_neutral, c_scale = 128 * scale, 224 * scale   # c_scale=896 for 10-bit

        y = np.clip(Y * y_range + y_black,       y_black,              y_black + y_range).round().astype(np.uint16)
        u = np.clip(Cb_ds * c_scale + c_neutral, c_neutral - c_scale/2, c_neutral + c_scale/2).round().astype(np.uint16)
        v = np.clip(Cr_ds * c_scale + c_neutral, c_neutral - c_scale/2, c_neutral + c_scale/2).round().astype(np.uint16)

        return y, u, v

    def _to_tensor(self, bgr):
        """HWC float32 → NCHW tensor on self.device (half if use_half)."""
        t = torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return t.half() if self.use_half else t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_model(self, base_bgr, prv_bgr, nxt_bgr):
        """Single forward pass; returns float32 HWC numpy array."""
        sr_t = self.model(
            self._to_tensor(base_bgr),
            self._to_tensor(prv_bgr),
            self._to_tensor(nxt_bgr),
        )
        return sr_t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    def _upscale_rgb_tiled(self, base_bgr, prv_bgr, nxt_bgr):
        """
        Process the frame in non-overlapping tiles to stay within GPU memory.

        tile_h / tile_w are in base (LR) coordinates.
        overlap pixels of context are added on each side and discarded after
        inference to soften tile-boundary artifacts.
        """
        H_in, W_in = base_bgr.shape[:2]
        s = self.scale
        H_out, W_out = H_in * s, W_in * s
        output = np.zeros((H_out, W_out, 3), dtype=np.float32)

        th = self.tile_h
        tw = self.tile_w
        ov = self.tile_overlap

        y = 0
        while y < H_in:
            x = 0
            while x < W_in:
                # Padded input bounds (clamped to image)
                iy0 = max(0, y - ov)
                iy1 = min(H_in, y + th + ov)
                ix0 = max(0, x - ov)
                ix1 = min(W_in, x + tw + ov)

                # Corresponding HR bounds for prv/nxt
                oy0, oy1 = iy0 * s, iy1 * s
                ox0, ox1 = ix0 * s, ix1 * s

                sr_tile = self._run_model(
                    base_bgr[iy0:iy1, ix0:ix1],
                    prv_bgr[oy0:oy1, ox0:ox1],
                    nxt_bgr[oy0:oy1, ox0:ox1],
                )
                torch.cuda.empty_cache()

                # Valid (non-padded) region inside this tile's output
                vy0 = (y - iy0) * s
                vx0 = (x - ix0) * s
                tile_out_h = min(th, H_in - y) * s
                tile_out_w = min(tw, W_in - x) * s

                output[y * s: y * s + tile_out_h,
                       x * s: x * s + tile_out_w] = \
                    sr_tile[vy0: vy0 + tile_out_h, vx0: vx0 + tile_out_w]

                x += tw
            y += th

        return np.clip(output, 0.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def upscale_rgb(self, y, u, v, prv_y, prv_u, prv_v, nxt_y, nxt_u, nxt_v):
        """
        Super-resolve the current LR frame using adjacent HR frames.

        All inputs are 10-bit YUV420 planes.
        Returns (sr_y, sr_u, sr_v) at 2× resolution as uint16.
        """
        base_bgr = self._yuv420_to_bgr(y, u, v)
        prv_bgr  = self._yuv420_to_bgr(prv_y, prv_u, prv_v)
        nxt_bgr  = self._yuv420_to_bgr(nxt_y, nxt_u, nxt_v)

        if self.tile_h is not None and self.tile_w is not None:
            sr_bgr = self._upscale_rgb_tiled(base_bgr, prv_bgr, nxt_bgr)
        else:
            sr_bgr = np.clip(self._run_model(base_bgr, prv_bgr, nxt_bgr), 0.0, 1.0).astype(np.float32)

        return self._bgr_to_yuv420(sr_bgr)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def yuv420_to_bgr_uint8(y, u, v, max_value=1023):
        """
        Convert 10-bit YUV420 planes to an 8-bit BGR image (H, W, 3) uint8
        suitable for cv2.imwrite / PNG saving.
        Uses BT.709 limited-range to match ffmpeg default output.
        """
        H, W = y.shape
        scale = (max_value + 1) / 256.0   # 4.0 for 10-bit, 1.0 for 8-bit
        y_black  = 16  * scale             # 64  for 10-bit
        y_range  = 219 * scale             # 876 for 10-bit
        c_neutral = 128 * scale            # 512 for 10-bit
        c_scale   = 224 * scale            # 896 for 10-bit → E'_Cb ∈ [-0.5,+0.5]

        Y = (y.astype(np.float32) - y_black) / y_range

        Cb = (cv2.resize(u.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
              - c_neutral) / c_scale
        Cr = (cv2.resize(v.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
              - c_neutral) / c_scale

        R = Y + 1.5748 * Cr
        G = Y - 0.1873 * Cb - 0.4681 * Cr
        B = Y + 1.8556 * Cb

        bgr = np.stack([B, G, R], axis=-1)
        return np.clip(bgr * 255.0, 0, 255).round().astype(np.uint8)
