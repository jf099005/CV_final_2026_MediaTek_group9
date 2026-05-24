import numpy as np
import os

def read_yuv420_10bit_frame(filepath, width, height, frame_idx):
    """
    Read one YUV420 10-bit frame.
    Assumes each sample is stored as uint16.
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + uv_size * 2

    offset = frame_idx * frame_size * 2  # uint16 = 2 bytes

    with open(filepath, "rb") as f:
        f.seek(offset)
        raw = f.read(frame_size * 2)

    if len(raw) < frame_size * 2:
        raise ValueError(f"Frame {frame_idx} out of range or incomplete.")

    data = np.frombuffer(raw, dtype=np.uint16)

    y = data[:y_size].reshape(height, width)
    u = data[y_size:y_size + uv_size].reshape(height // 2, width // 2)
    v = data[y_size + uv_size:y_size + uv_size * 2].reshape(height // 2, width // 2)

    return y, u, v


def write_yuv420_10bit_frame(filepath, y, u, v, append=True):
    mode = "ab" if append else "wb"

    with open(filepath, mode) as f:
        f.write(y.astype(np.uint16).tobytes())
        f.write(u.astype(np.uint16).tobytes())
        f.write(v.astype(np.uint16).tobytes())


def get_total_frames_yuv420_10bit(filepath, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + uv_size * 2
    frame_bytes = frame_size * 2

    file_size = os.path.getsize(filepath)
    return file_size // frame_bytes