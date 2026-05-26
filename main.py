import argparse
import subprocess
from tqdm import tqdm

from utils.ReadAndWrite import read_yuv420_10bit_frames, parse_yuv420_10bit, write_yuv420_10bit_frame, get_total_frames
from utils.GenerateFrame import generate_frame
from utils.YOnlySR import YOnlySR


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
        "--model_path",
        required=True,
        help="Path to trained EDSREnd2EndModel checkpoint (.pth)"
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

    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Super-resolution scale factor"
    )

    parser.add_argument(
        "--bit_depth",
        type=int,
        default=10,
        help="Bit depth of YUV files"
    )

    parser.add_argument(
        "--output_video",
        default=None,
        help="If given, convert the output YUV to a video file at this path (e.g. out.mp4)"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used when encoding the output video (default: 30)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    sr_model = YOnlySR(
        model_path=args.model_path,
        scale=args.scale,
        bit_depth=args.bit_depth,
    )

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
            total=num_enh_frames,
            desc="Generating 4K odd frames"
        ):
            try:
                _, raw_e_next = next(enh_reader)
            except StopIteration:
                print("No next enhancement frame. Stop.")
                break

            b_y, b_u, b_v = parse_yuv420_10bit(raw_b, base_width, base_height)

            e_prev_y, e_prev_u, e_prev_v = parse_yuv420_10bit(raw_e_prev, enh_width, enh_height)
            e_next_y, e_next_u, e_next_v = parse_yuv420_10bit(raw_e_next, enh_width, enh_height)

            pred_y, pred_u, pred_v = generate_frame(
                sr_model,
                b_y, b_u, b_v,
                e_prev_y, e_prev_u, e_prev_v,
                e_next_y, e_next_u, e_next_v,
            )

            write_yuv420_10bit_frame(out_f, pred_y, pred_u, pred_v)

            raw_e_prev = raw_e_next

    print("Done.")
    print("Output saved to:")
    print(output_path)

    if args.output_video:
        print(f"Converting YUV to video: {args.output_video}")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "yuv420p10le",
            "-video_size", f"{enh_width}x{enh_height}",
            "-framerate", str(args.fps),
            "-i", output_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            args.output_video,
        ]
        subprocess.run(cmd, check=True)
        print("Video saved to:")
        print(args.output_video)


if __name__ == "__main__":
    main()
