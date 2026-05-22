from utils.Resize import resize_yuv420_10bit
from utils.Wraping import backward_warp_plane_with_mask
from utils.RAFTFlow import RAFTFlowEstimator
import cv2
import numpy as np

raft_estimator = RAFTFlowEstimator()

def fuse_sources_with_mask(base, prev, next_, mask_prev, mask_next):
    """
    Fuse one plane from three sources:

        base:  upsampled B[t]
        prev:  warped E[t-1]
        next_: warped E[t+1]

    base always exists.
    prev / next only contribute where their masks are valid.

    Weight:
        base = 0.2
        prev = 0.4 if valid
        next = 0.4 if valid
    """

    base_f = base.astype(np.float32)
    prev_f = prev.astype(np.float32)
    next_f = next_.astype(np.float32)

    out = 0.2 * base_f
    weight = np.full(base.shape, 0.2, dtype=np.float32)

    if mask_prev is not None:
        out[mask_prev] += 0.4 * prev_f[mask_prev]
        weight[mask_prev] += 0.4

    if mask_next is not None:
        out[mask_next] += 0.4 * next_f[mask_next]
        weight[mask_next] += 0.4

    out = out / weight

    out = np.clip(np.rint(out), 0, 1023).astype(np.uint16)

    return out

def resize_flow(flow, out_width, out_height):
    in_height, in_width = flow.shape[:2]

    scale_x = out_width / in_width
    scale_y = out_height / in_height

    flow_resized = cv2.resize(
        flow,
        (out_width, out_height),
        interpolation=cv2.INTER_LINEAR
    )

    flow_resized[..., 0] *= scale_x
    flow_resized[..., 1] *= scale_y
    return flow_resized


def generate_frame(b_y, b_u, b_v
            ,e_prev_y, e_prev_u, e_prev_v
            ,e_next_y, e_next_u, e_next_v):
        
        b_4k_y, b_4k_u, b_4k_v = resize_yuv420_10bit(b_y, b_u, b_v, 3840, 2160, interpolation=cv2.INTER_CUBIC)
        e_prev_fhd_y, e_prev_fhd_u, e_prev_fhd_v = resize_yuv420_10bit(e_prev_y, e_prev_u, e_prev_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        e_next_fhd_y, e_next_fhd_u, e_next_fhd_v = resize_yuv420_10bit(e_next_y, e_next_u, e_next_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        
        raft_w = 960
        raft_h = 544

        b_raft_y, _, _ = resize_yuv420_10bit(
            b_y, b_u, b_v,
            raft_w, raft_h,
            interpolation=cv2.INTER_AREA
        )

        e_prev_raft_y, _, _ = resize_yuv420_10bit(
            e_prev_y, e_prev_u, e_prev_v,
                raft_w, raft_h,
                interpolation=cv2.INTER_AREA
            )

        e_next_raft_y, _, _ = resize_yuv420_10bit(
            e_next_y, e_next_u, e_next_v,
            raft_w, raft_h,
            interpolation=cv2.INTER_AREA
            )

        flow_curr2prev = raft_estimator.compute_flow(
            b_raft_y,
            e_prev_raft_y
            )

        flow_curr2next = raft_estimator.compute_flow(
            b_raft_y,
            e_next_raft_y
            )

        flow_curr2prev_4k = resize_flow(
            flow_curr2prev,
            out_width=3840,
            out_height=2160
            )

        flow_curr2next_4k = resize_flow(
            flow_curr2next,
            out_width=3840,
            out_height=2160
            )       

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


