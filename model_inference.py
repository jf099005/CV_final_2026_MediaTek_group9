import argparse
import os

import cv2
from tqdm import tqdm

from utils.ReadAndWrite import read_yuv420_10bit_frames, parse_yuv420_10bit, write_yuv420_10bit_frame, get_total_frames
from utils.RGBBasedSR import RGBBasedSR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "RGB_SR", "checkpoints_rgb", "best.pth")
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "RGB_SR", "models", "model_config.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RGB SR model inference only (no RAFT / flow fusion)"
    )
    parser.add_argument("--base",         required=True, help="Path to FHD base-layer YUV file")
    parser.add_argument("--enhancement",  required=True, help="Path to 4K enhancement-layer YUV file")
    parser.add_argument("--output",       required=True, help="Path to output 4K YUV file")
    parser.add_argument("--png_output_dir", default=None,
                        help="Directory to save per-frame PNG outputs (default: <output_basename>_frames/)")
    parser.add_argument("--base_width",   type=int, default=1920)
    parser.add_argument("--base_height",  type=int, default=1080)
    parser.add_argument("--enh_width",    type=int, default=3840)
    parser.add_argument("--enh_height",   type=int, default=2160)
    parser.add_argument("--bit_depth",    type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {MODEL_PATH}")
    sr_model = RGBBasedSR(model_path=MODEL_PATH, bit_depth=args.bit_depth, config_path=CONFIG_PATH)

    png_dir = args.png_output_dir
    if png_dir is None:
        png_dir = os.path.splitext(args.output)[0] + "_frames"
    os.makedirs(png_dir, exist_ok=True)
    print(f"PNG frames will be saved to: {png_dir}")

    max_value      = (1 << args.bit_depth) - 1
    num_enh_frames = get_total_frames(args.enhancement, args.enh_width, args.enh_height)

    base_reader = read_yuv420_10bit_frames(args.base, args.base_width, args.base_height)
    enh_reader  = read_yuv420_10bit_frames(args.enhancement, args.enh_width, args.enh_height)

    _, raw_e_prev = next(enh_reader)

    with open(args.output, "wb") as out_f:
        for base_idx, raw_b in tqdm(base_reader, total=num_enh_frames, desc="Inference"):
            try:
                _, raw_e_next = next(enh_reader)
            except StopIteration:
                print("No next enhancement frame. Stop.")
                break

            b_y,      b_u,      b_v      = parse_yuv420_10bit(raw_b,      args.base_width,  args.base_height)
            e_prev_y, e_prev_u, e_prev_v = parse_yuv420_10bit(raw_e_prev, args.enh_width,   args.enh_height)
            e_next_y, e_next_u, e_next_v = parse_yuv420_10bit(raw_e_next, args.enh_width,   args.enh_height)

            pred_y, pred_u, pred_v = sr_model.upscale_rgb(
                b_y, b_u, b_v,
                e_prev_y, e_prev_u, e_prev_v,
                e_next_y, e_next_u, e_next_v,
            )

            write_yuv420_10bit_frame(out_f, pred_y, pred_u, pred_v)

            bgr_u8 = RGBBasedSR.yuv420_to_bgr_uint8(pred_y, pred_u, pred_v, max_value)
            cv2.imwrite(os.path.join(png_dir, f"frame_{base_idx:05d}.png"), bgr_u8)

            raw_e_prev = raw_e_next

    print("Done.")
    print(f"YUV output : {args.output}")
    print(f"PNG frames : {png_dir}")


if __name__ == "__main__":
    main()
