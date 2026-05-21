from utils.Resize import resize_yuv420_10bit
from utils.OpticalFlow import optical_flow, upscale_flow_2x, forward_warp_yuv420
import cv2
import numpy as np

def fuse_three_planes_with_mask(base, prev, next_, mask_prev, mask_next):
    """
        base weight = 0.5
        prev weight = 0.25 if valid
        next weight = 0.25 if valid
    """
    base_f = base.astype(np.float32)
    prev_f = prev.astype(np.float32)
    next_f = next_.astype(np.float32)

    out = 0.5 * base_f
    weight = np.full(base.shape, 0.5, dtype=np.float32)

    out[mask_prev] += 0.25 * prev_f[mask_prev]
    weight[mask_prev] += 0.25

    out[mask_next] += 0.25 * next_f[mask_next]
    weight[mask_next] += 0.25

    out = out / weight
    out = np.clip(np.rint(out), 0, 1023).astype(np.uint16)

    return out

def fuse_yuv420_frames_with_mask(
    B_up_y, B_up_u, B_up_v,
    warped_prev_y, warped_prev_u, warped_prev_v,
    warped_next_y, warped_next_u, warped_next_v,
    mask_prev_y, mask_prev_u, mask_prev_v,
    mask_next_y, mask_next_u, mask_next_v
):
    P_y = fuse_three_planes_with_mask(
        B_up_y, warped_prev_y, warped_next_y,
        mask_prev_y, mask_next_y
    )

    P_u = fuse_three_planes_with_mask(
        B_up_u, warped_prev_u, warped_next_u,
        mask_prev_u, mask_next_u
    )

    P_v = fuse_three_planes_with_mask(
        B_up_v, warped_prev_v, warped_next_v,
        mask_prev_v, mask_next_v
    )

    return P_y, P_u, P_v

def generate_frame(b_y, b_u, b_v
            ,e_prev_y, e_prev_u, e_prev_v
            ,e_next_y, e_next_u, e_next_v):
        
        b_4k_y, b_4k_u, b_4k_v = resize_yuv420_10bit(b_y, b_u, b_v, 3840, 2160, interpolation=cv2.INTER_CUBIC)
        e_prev_fhd_y, e_prev_fhd_u, e_prev_fhd_v = resize_yuv420_10bit(e_prev_y, e_prev_u, e_prev_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        e_next_fhd_y, e_next_fhd_u, e_next_fhd_v = resize_yuv420_10bit(e_next_y, e_next_u, e_next_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        flow_prev = optical_flow(e_prev_fhd_y, b_y)
        flow_next = optical_flow(e_next_fhd_y, b_y)
        flow_prev_4k = upscale_flow_2x(flow_prev)
        flow_next_4k = upscale_flow_2x(flow_next)
        
        warped_prev_y, warped_prev_u, warped_prev_v,mask_prev_y, mask_prev_u, mask_prev_v = forward_warp_yuv420(
            e_prev_y, e_prev_u, e_prev_v,
            flow_prev_4k
        )
        warped_next_y, warped_next_u, warped_next_v,mask_next_y, mask_next_u, mask_next_v = forward_warp_yuv420(
            e_next_y, e_next_u, e_next_v,
            flow_next_4k
        )
        P_y, P_u, P_v = fuse_yuv420_frames_with_mask(
            b_4k_y, b_4k_u, b_4k_v,
            warped_prev_y, warped_prev_u, warped_prev_v,
            warped_next_y, warped_next_u, warped_next_v,
            mask_prev_y, mask_prev_u, mask_prev_v,
            mask_next_y, mask_next_u, mask_next_v
        )
        return P_y, P_u, P_v


