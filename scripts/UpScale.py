import numpy as np
import cv2


def read_yuv420_10bit_frames(filepath, width, height):
    frame_size_samples = width * height * 3 // 2
    frame_size_bytes = frame_size_samples * 2  # uint16 = 2 bytes

    with open(filepath, "rb") as f:
        frame_idx = 0

        while True:
            raw = f.read(frame_size_bytes)

            if len(raw) < frame_size_bytes:
                break

            yield frame_idx, raw
            frame_idx += 1


def parse_yuv420_10bit(raw, width, height):
    y_size = width * height
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height

    yuv = np.frombuffer(raw, dtype=np.uint16)

    y = yuv[0:y_size].reshape((height, width))
    u = yuv[y_size:y_size + uv_size].reshape((uv_height, uv_width))
    v = yuv[y_size + uv_size:y_size + uv_size * 2].reshape((uv_height, uv_width))

    return y, u, v


def upscale_yuv420_10bit(y, u, v, out_width, out_height):
    # Y plane: 1920x1080 -> 3840x2160
    y_4k = cv2.resize(
        y,
        (out_width, out_height),
        interpolation=cv2.INTER_NEAREST
    )

    # U/V plane: 960x540 -> 1920x1080
    u_4k = cv2.resize(
        u,
        (out_width // 2, out_height // 2),
        interpolation=cv2.INTER_NEAREST
    )

    v_4k = cv2.resize(
        v,
        (out_width // 2, out_height // 2),
        interpolation=cv2.INTER_NEAREST
    )

    return y_4k, u_4k, v_4k


def write_yuv420_10bit_frame(f, y, u, v):
    f.write(y.astype(np.uint16).tobytes())
    f.write(u.astype(np.uint16).tobytes())
    f.write(v.astype(np.uint16).tobytes())


input_path = r"C:\Users\User\OneDrive\projects\CV_final_2026_MediaTek_group9\bitstream\base\odd_H2_H3_AMS05_27_0_5.layer0.yuv"
output_path = r"C:\Users\User\OneDrive\projects\CV_final_2026_MediaTek_group9\results\odd_H2_H3_AMS05_27_0_5_4k_10bit.yuv"

in_width = 1920
in_height = 1080

out_width = 3840
out_height = 2160

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