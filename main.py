import argparse
import os
from tqdm import tqdm

from utils.SimpleUpScale import upscale_yuv420_10bit
from utils.ReadAndWrite import read_yuv420_10bit_frames, parse_yuv420_10bit, write_yuv420_10bit_frame, get_total_frames

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate missing 4K odd frames using FHD base layer and adjacent 4K enhancement frames"
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Path to FHD base-layer YUV file"
    )

    parser.add_argument(
        "--enhancement",
        required=True,
        help="Path to 4K enhancement-layer YUV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output generated 4K YUV file"
    )

    parser.add_argument(
        "--base_width",
        type=int,
        default=1920,
        help="Base-layer width"
    )

    parser.add_argument(
        "--base_height",
        type=int,
        default=1080,
        help="Base-layer height"
    )

    parser.add_argument(
        "--enh_width",
        type=int,
        default=3840,
        help="Enhancement-layer width"
    )

    parser.add_argument(
        "--enh_height",
        type=int,
        default=2160,
        help="Enhancement-layer height"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    base_path = args.base
    enhancement_path = args.enhancement
    output_path = args.output

    base_width = args.base_width
    base_height = args.base_height

    enh_width = args.enh_width
    enh_height = args.enh_height

    num_base_frames = get_total_frames(base_path, base_width, base_height)
    num_enh_frames = get_total_frames(enhancement_path, enh_width, enh_height)

    base_reader = read_yuv420_10bit_frames(
        base_path,
        base_width,
        base_height
    )

    enh_reader = read_yuv420_10bit_frames(
        enhancement_path,
        enh_width,
        enh_height
    )
    _, raw_e_prev = next(enh_reader)

    with open(output_path, "wb") as out_f:
        for base_idx, raw_b in tqdm(
            base_reader,
            total=num_base_frames,
            desc="Generating 4K odd frames"
        ):
            # 讀下一張 enhancement frame，作為後一張 4K frame
            try:
                _, raw_e_next = next(enh_reader)
            except StopIteration:
                print("No next enhancement frame. Stop.")
                break

            # Parse current FHD base frame
            b_y, b_u, b_v = parse_yuv420_10bit(
                raw_b,
                base_width,
                base_height
            )

            # Parse previous 4K enhancement frame
            e_prev_y, e_prev_u, e_prev_v = parse_yuv420_10bit(
                raw_e_prev,
                enh_width,
                enh_height
            )

            # Parse next 4K enhancement frame
            e_next_y, e_next_u, e_next_v = parse_yuv420_10bit(
                raw_e_next,
                enh_width,
                enh_height
            )

            # 現在你已經有三個 input：
            #
            # B[t]      = b_y, b_u, b_v
            # E[t - 1] = e_prev_y, e_prev_u, e_prev_v
            # E[t + 1] = e_next_y, e_next_u, e_next_v
            #
            # 先暫時用 simple upscale 當 output baseline
            pred_y, pred_u, pred_v = upscale_yuv420_10bit(
                b_y,
                b_u,
                b_v,
                enh_width,
                enh_height
            )

            write_yuv420_10bit_frame(
                out_f,
                pred_y,
                pred_u,
                pred_v
            )

            # 下一個 base frame 的 previous enhancement frame
            raw_e_prev = raw_e_next

    print("Done.")
    print("Output saved to:")
    print(output_path)


if __name__ == "__main__":
    main()