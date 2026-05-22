from utils.Resize import resize_yuv420_10bit
from utils.OpticalFlow import optical_flow, upscale_flow_2x
from utils.Wraping import backward_warp_plane_with_mask
import cv2
import numpy as np

def fuse_sources_with_mask(base, prev, next_, mask_prev, mask_next):
    """
    Fuse one plane from three sources:

        base:  upsampled B[t]
        prev:  warped E[t-1]
        next_: warped E[t+1]

    base always exists.
    prev / next only contribute where their masks are valid.

    Weight:
        base = 0.5
        prev = 0.25 if valid
        next = 0.25 if valid
    """

    base_f = base.astype(np.float32)
    prev_f = prev.astype(np.float32)
    next_f = next_.astype(np.float32)

    out = 0.5 * base_f
    weight = np.full(base.shape, 0.5, dtype=np.float32)

    if mask_prev is not None:
        out[mask_prev] += 0.25 * prev_f[mask_prev]
        weight[mask_prev] += 0.25

    if mask_next is not None:
        out[mask_next] += 0.25 * next_f[mask_next]
        weight[mask_next] += 0.25

    out = out / weight

    out = np.clip(np.rint(out), 0, 1023).astype(np.uint16)

    return out

def generate_frame(b_y, b_u, b_v
            ,e_prev_y, e_prev_u, e_prev_v
            ,e_next_y, e_next_u, e_next_v):
        
        b_4k_y, b_4k_u, b_4k_v = resize_yuv420_10bit(b_y, b_u, b_v, 3840, 2160, interpolation=cv2.INTER_CUBIC)
        e_prev_fhd_y, e_prev_fhd_u, e_prev_fhd_v = resize_yuv420_10bit(e_prev_y, e_prev_u, e_prev_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        e_next_fhd_y, e_next_fhd_u, e_next_fhd_v = resize_yuv420_10bit(e_next_y, e_next_u, e_next_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        
        flow_curr2prev = optical_flow(b_y, e_prev_fhd_y)
        flow_curr2next = optical_flow(b_y, e_next_fhd_y)

        flow_curr2prev_4k = upscale_flow_2x(flow_curr2prev)
        flow_curr2next_4k = upscale_flow_2x(flow_curr2next)

        warped_prev_y, mask_prev_y = backward_warp_plane_with_mask(
            e_prev_y,
            flow_curr2prev_4k
        )

        warped_next_y, mask_next_y = backward_warp_plane_with_mask(
            e_next_y,
            flow_curr2next_4k
        )

        P_y = fuse_sources_with_mask(
            b_4k_y,
            warped_prev_y,
            warped_next_y,
            mask_prev_y,
            mask_next_y
        )

        P_u = b_4k_u
        P_v = b_4k_v
        return P_y, P_u, P_v


