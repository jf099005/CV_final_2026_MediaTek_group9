import os
import numpy as np
import cv2

def read_yuv420_10bit_frames(filepath, width, height):
    # 10-bit packed as uint16 (little-endian), so 2 bytes per sample
    frame_size_samples = width * height * 3 // 2
    frame_size_bytes = frame_size_samples * 2  # uint16

    with open(filepath, "rb") as f:
        frame_idx = 0
        while True:
            raw = f.read(frame_size_bytes)
            if len(raw) < frame_size_bytes:
                break
            yield frame_idx, raw
            frame_idx += 1

filepath = r"C:\Users\User\OneDrive\projects\CV_final_2026_MediaTek_group9\bitstream\base\odd_H2_H3_AMS05_27_0_5.layer0.yuv"

width = 1920
height = 1080
target_idx = 50

for frame_idx, raw in read_yuv420_10bit_frames(filepath, width, height):
    if frame_idx == target_idx:
        # Parse as uint16, extract 10-bit values, shift down to 8-bit
        yuv16 = np.frombuffer(raw, dtype=np.uint16)
        yuv8 = (yuv16 >> 2).astype(np.uint8)  # 10-bit → 8-bit by dropping 2 LSBs

        yuv_frame = yuv8.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        cv2.imwrite(f"frame_{frame_idx:04d}.png", bgr)
        print(f"Saved frame_{frame_idx:04d}.png")
        break