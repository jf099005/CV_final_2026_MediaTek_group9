import argparse

from utils.SimpleUpScale import upscale_yuv420_10bit
from utils.ReadAndWrite import read_yuv420_10bit_frames, parse_yuv420_10bit, write_yuv420_10bit_frame

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upscale 10-bit YUV420 video to 4K 10-bit YUV420"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input YUV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output YUV file"
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Input video width"
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Input video height"
    )

    parser.add_argument(
        "--out_width",
        type=int,
        default=3840,
        help="Output video width"
    )

    parser.add_argument(
        "--out_height",
        type=int,
        default=2160,
        help="Output video height"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output

    in_width = args.width
    in_height = args.height

    out_width = args.out_width
    out_height = args.out_height

    with open(output_path, "wb") as out_f:
        for frame_idx, raw in read_yuv420_10bit_frames(input_path, in_width, in_height):
            y, u, v = parse_yuv420_10bit(raw, in_width, in_height)

            y_4k, u_4k, v_4k = upscale_yuv420_10bit(
                y, u, v,
                out_width,
                out_height
            )

            write_yuv420_10bit_frame(out_f, y_4k, u_4k, v_4k)

            print(f"Processed frame {frame_idx}")

    print("Done.")
    print(f"Saved 4K 10-bit YUV420 video to:")
    print(output_path)

if __name__ == "__main__":
    main()