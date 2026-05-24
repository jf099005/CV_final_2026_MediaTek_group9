import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.edsr_small import EDSRSmall
from utils.yuv_io import read_yuv420_10bit_frame, write_yuv420_10bit_frame


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_yuv", required=True)
    parser.add_argument("--output_yuv", required=True)
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--in_width", type=int, default=1920)
    parser.add_argument("--in_height", type=int, default=1080)
    parser.add_argument("--out_width", type=int, default=3840)
    parser.add_argument("--out_height", type=int, default=2160)

    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--bit_depth", type=int, default=10)

    return parser.parse_args()


@torch.no_grad()
def infer():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    max_value = (1 << args.bit_depth) - 1

    model = EDSRSmall(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=8,
        scale=args.scale,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if os.path.exists(args.output_yuv):
        os.remove(args.output_yuv)

    for frame_idx in tqdm(range(args.num_frames)):
        y, u, v = read_yuv420_10bit_frame(
            args.input_yuv,
            args.in_width,
            args.in_height,
            frame_idx,
        )

        y_float = y.astype(np.float32) / max_value
        y_tensor = torch.from_numpy(y_float).unsqueeze(0).unsqueeze(0).to(device)

        pred_y = model(y_tensor)

        pred_y = pred_y.squeeze(0).squeeze(0).cpu().numpy()
        pred_y = np.clip(pred_y * max_value, 0, max_value).round().astype(np.uint16)

        # U/V 先用 resize baseline
        sr_u = cv2.resize(
            u,
            (args.out_width // 2, args.out_height // 2),
            interpolation=cv2.INTER_CUBIC,
        )

        sr_v = cv2.resize(
            v,
            (args.out_width // 2, args.out_height // 2),
            interpolation=cv2.INTER_CUBIC,
        )

        sr_u = np.clip(sr_u, 0, max_value).round().astype(np.uint16)
        sr_v = np.clip(sr_v, 0, max_value).round().astype(np.uint16)

        write_yuv420_10bit_frame(
            args.output_yuv,
            pred_y,
            sr_u,
            sr_v,
            append=True,
        )


if __name__ == "__main__":
    infer()