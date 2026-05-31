import argparse
import json
import os
import subprocess

import cv2
from tqdm import tqdm

from utils.ReadAndWrite import read_yuv420_10bit_frames, parse_yuv420_10bit, write_yuv420_10bit_frame, get_total_frames
from utils.GenerateFrame import generate_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate missing 4K odd frames using FHD base layer and adjacent 4K enhancement frames"
    )

    parser.add_argument("--base", required=True, help="Path to FHD base-layer YUV file")
    parser.add_argument("--enhancement", required=True, help="Path to 4K enhancement-layer YUV file")
    parser.add_argument("--output", required=True, help="Path to output generated 4K YUV file")

    parser.add_argument("--model_path", required=False, default=None,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--config_path", required=False, default=None,
                        help="Path to model_config.json (auto-detected from model type if omitted)")

    parser.add_argument("--naive", action="store_true",
                        help="Skip SR model and upscale base frame by linear interpolation")

    parser.add_argument("--base_width",  type=int, default=1920)
    parser.add_argument("--base_height", type=int, default=1080)
    parser.add_argument("--enh_width",   type=int, default=3840)
    parser.add_argument("--enh_height",  type=int, default=2160)
    parser.add_argument("--bit_depth",   type=int, default=10)

    parser.add_argument("--output_video", default=None,
                        help="If given, convert the output YUV to a video file at this path (e.g. out.mp4)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate used when encoding the output video (default: 30)")

    parser.add_argument("--only_edsr", action="store_true",
                        help="Run SR only, skip flow-based fusion (ablation study)")
    parser.add_argument("--hr_input_type", type=str, default="warped",
                        choices=["warped", "prv and nxt"],
                        help="HR reference type for Y-only model fusion")

    parser.add_argument("--png_output_dir", default=None,
                        help="Directory to save per-frame PNG outputs. "
                             "Defaults to '<output_basename>_frames/' next to the YUV output.")

    return parser.parse_args()


def _load_sr_model(model_path, bit_depth, config_path):
    """
    Auto-detect RGB vs Y-only model from the config's in_channels field.
    Returns an RGBBasedSR or YOnlySR instance.
    """
    # Resolve config path: if not given, probe both model directories
    if config_path is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        rgb_cfg = os.path.join(project_root, "RGB_SR", "models", "model_config.json")
        yuv_cfg = os.path.join(project_root, "YUV_SR", "models", "model_config.json")
        config_path = rgb_cfg if os.path.exists(rgb_cfg) else yuv_cfg

    with open(config_path) as f:
        cfg = json.load(f)

    in_ch = cfg.get("model", {}).get("in_channels", 1)

    model_type = cfg.get("model", {}).get("type", "edsr")
    if in_ch == 3 or model_type == "carn":
        from utils.RGBBasedSR import RGBBasedSR
        print(f"Using RGB model  (type={model_type}, in_channels={in_ch}): {config_path}")
        return RGBBasedSR(model_path=model_path, bit_depth=bit_depth, config_path=config_path)
    else:
        from utils.YOnlySR import YOnlySR
        print(f"Using Y-only model (in_channels={in_ch}): {config_path}")
        return YOnlySR(model_path=model_path, bit_depth=bit_depth, config_path=config_path)


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    if args.naive:
        sr_model = None
    else:
        # CARN loads its backbone from carn_pretrained_path in the config,
        # so model_path is optional for CARN (used only for a full combined ckpt).
        cfg_check = {}
        if args.config_path and os.path.exists(args.config_path):
            import json as _json
            with open(args.config_path) as _f:
                cfg_check = _json.load(_f)
        is_carn = cfg_check.get("model", {}).get("type") == "carn"
        if args.model_path is None and not is_carn:
            raise ValueError("--model_path is required unless --naive is set")
        sr_model = _load_sr_model(args.model_path, args.bit_depth, args.config_path)

    # ------------------------------------------------------------------ #
    # PNG output directory
    # ------------------------------------------------------------------ #
    png_dir = args.png_output_dir
    if png_dir is None:
        base_name = os.path.splitext(args.output)[0]
        png_dir = base_name + "_frames"
    os.makedirs(png_dir, exist_ok=True)
    print(f"PNG frames will be saved to: {png_dir}")

    # ------------------------------------------------------------------ #
    # YUV readers
    # ------------------------------------------------------------------ #
    base_width  = args.base_width
    base_height = args.base_height
    enh_width   = args.enh_width
    enh_height  = args.enh_height
    max_value   = (1 << args.bit_depth) - 1

    num_enh_frames = get_total_frames(args.enhancement, enh_width, enh_height)

    base_reader = read_yuv420_10bit_frames(args.base, base_width, base_height)
    enh_reader  = read_yuv420_10bit_frames(args.enhancement, enh_width, enh_height)

    _, raw_e_prev = next(enh_reader)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    with open(args.output, "wb") as out_f:
        for base_idx, raw_b in tqdm(
            base_reader,
            total=num_enh_frames,
            desc="Generating 4K odd frames",
        ):
            try:
                _, raw_e_next = next(enh_reader)
            except StopIteration:
                print("No next enhancement frame. Stop.")
                break

            b_y, b_u, b_v = parse_yuv420_10bit(raw_b,     base_width, base_height)
            e_prev_y, e_prev_u, e_prev_v = parse_yuv420_10bit(raw_e_prev, enh_width, enh_height)
            e_next_y, e_next_u, e_next_v = parse_yuv420_10bit(raw_e_next, enh_width, enh_height)

            pred_y, pred_u, pred_v = generate_frame(
                sr_model,
                b_y, b_u, b_v,
                e_prev_y, e_prev_u, e_prev_v,
                e_next_y, e_next_u, e_next_v,
                only_edsr=args.only_edsr,
                hr_input_type=args.hr_input_type,
            )

            # ---- Save YUV (original format) ----
            write_yuv420_10bit_frame(out_f, pred_y, pred_u, pred_v)

            # ---- Save PNG ----
            from utils.RGBBasedSR import RGBBasedSR
            bgr_u8 = RGBBasedSR.yuv420_to_bgr_uint8(pred_y, pred_u, pred_v, max_value)
            png_path = os.path.join(png_dir, f"frame_{base_idx:05d}.png")
            cv2.imwrite(png_path, bgr_u8)

            raw_e_prev = raw_e_next

    print("Done.")
    print(f"YUV output : {args.output}")
    print(f"PNG frames : {png_dir}")

    # ------------------------------------------------------------------ #
    # Optional video encode
    # ------------------------------------------------------------------ #
    if args.output_video:
        print(f"Converting YUV to video: {args.output_video}")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "yuv420p10le",
            "-video_size", f"{enh_width}x{enh_height}",
            "-framerate", str(args.fps),
            "-i", args.output,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            args.output_video,
        ]
        subprocess.run(cmd, check=True)
        print(f"Video saved to: {args.output_video}")


if __name__ == "__main__":
    main()
