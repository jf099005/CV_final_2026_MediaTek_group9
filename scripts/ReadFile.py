import numpy as np
import cv2


def read_yuv420_raw_frames(filepath, width, height):
    frame_size = width * height * 3 // 2

    with open(filepath, "rb") as f:
        frame_idx = 0

        while True:
            raw = f.read(frame_size)

            if len(raw) < frame_size:
                break

            yield frame_idx, raw

            frame_idx += 1


filepath = r"C:\Users\User\OneDrive\projects\CV_final_2026_MediaTek_group9\bitstream\base\odd_H2_H3_AMS05_27_0_5.layer0.yuv"

width = 1920
height = 1080
target_idx = 50

for frame_idx, raw in read_yuv420_raw_frames(filepath, width, height):
    if frame_idx == target_idx:
        # 1. 存成單一 YUV frame
        with open(f"frame_{frame_idx:04d}.yuv", "wb") as out:
            out.write(raw)

        # 2. 轉成 PNG 檢查
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv_frame = yuv.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        cv2.imwrite(f"frame_{frame_idx:04d}.png", bgr)

        print(f"Saved frame_{frame_idx:04d}.yuv")
        print(f"Saved frame_{frame_idx:04d}.png")
        break