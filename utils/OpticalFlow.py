import numpy as np
import cv2

def y10_to_uint8(y):
    """
    Convert 10-bit Y plane to 8-bit for optical flow.
    """
    y = np.clip(y, 0, 1023)
    y8 = (y / 4).astype(np.uint8)
    return y8

def optical_flow(prev_y, curr_y):
    """
    Calculate optical flow from prev_y to curr_y.
    """

    prev_y = y10_to_uint8(prev_y)
    curr_y = y10_to_uint8(curr_y)

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_y,
        next=curr_y,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow

def upscale_flow_2x(flow_fhd):
    """
    Upscale optical flow from FHD to 4K.

    Input:
        flow_fhd: shape (H, W, 2), e.g. (1080, 1920, 2)

    Output:
        flow_4k: shape (2H, 2W, 2), e.g. (2160, 3840, 2)

    Important:
        flow values must also be multiplied by 2,
        because flow is measured in pixel displacement.
    """

    h, w = flow_fhd.shape[:2]

    flow_4k = cv2.resize(
        flow_fhd,
        (w * 2, h * 2),
        interpolation=cv2.INTER_LINEAR
    )

    flow_4k[..., 0] *= 2.0
    flow_4k[..., 1] *= 2.0

    return flow_4k
