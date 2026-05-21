import numpy as np
import os

def read_yuv420_10bit_frames(filepath, width, height):
    frame_size_samples = width * height * 3 // 2
    frame_size_bytes = frame_size_samples * 2  

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


def write_yuv420_10bit_frame(f, y, u, v):
    f.write(y.astype(np.uint16).tobytes())
    f.write(u.astype(np.uint16).tobytes())
    f.write(v.astype(np.uint16).tobytes())

def get_total_frames(filepath, width, height):
    frame_size_samples = width * height * 3 // 2
    frame_size_bytes = frame_size_samples * 2  # 10-bit stored as uint16

    file_size = os.path.getsize(filepath)
    total_frames = file_size // frame_size_bytes

    return total_frames