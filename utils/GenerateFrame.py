from utils.Resize import resize_yuv420_10bit
from utils.Wraping import backward_warp_plane_with_mask
from utils.RAFTFlow import RAFTFlowEstimator
from utils.YOnlySR import YOnlySR
import cv2
import numpy as np

raft_estimator = RAFTFlowEstimator()

def load_model(model_path, in_channels=1, out_channels=1, scale=2, bit_depth=10):
    sr_model = YOnlySR(
        model_path=model_path,
        scale=scale,
        bit_depth=bit_depth,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return sr_model

def upscale_yuv420_10bit_with_sr(model, y, u, v):
    y_4k, u_4k, v_4k = model.upscale_yuv420(y, u, v)
    return y_4k, u_4k, v_4k

import numpy as np


def fuse_sources_with_mask_adaptive(
    base,
    prev,
    next_,
    mask_prev=None,
    mask_next=None,
    bit_depth=10,
    base_weight=0.3,
    sigma=30.0,
    use_prev_next_consistency=True,
):
    """
    Adaptive fusion for one YUV plane.

    base:   upsampled B[t]
    prev:   warped E[t-1]
    next_:  warped E[t+1]

    mask_prev / mask_next:
        Boolean masks. True means this warped pixel is valid.

    Weight design:
        base always has a fixed minimum weight.
        prev / next weights depend on residual error:
            smaller |warped - base| -> larger weight
            larger  |warped - base| -> smaller weight

        Optional:
            if prev and next disagree strongly, reduce both temporal weights.
    """

    base_f = base.astype(np.float32)
    prev_f = prev.astype(np.float32)
    next_f = next_.astype(np.float32)

    max_val = (1 << bit_depth) - 1

    # Residual error to base
    err_prev = np.abs(prev_f - base_f)
    err_next = np.abs(next_f - base_f)

    # Convert residual error to reliability weights
    w_prev = np.exp(-err_prev / sigma)
    w_next = np.exp(-err_next / sigma)

    # Apply validity masks
    if mask_prev is not None:
        w_prev = w_prev * mask_prev.astype(np.float32)
    else:
        w_prev = np.zeros_like(base_f, dtype=np.float32)

    if mask_next is not None:
        w_next = w_next * mask_next.astype(np.float32)
    else:
        w_next = np.zeros_like(base_f, dtype=np.float32)

    # Optional prev-next consistency check
    if use_prev_next_consistency:
        err_pn = np.abs(prev_f - next_f)
        consistency = np.exp(-err_pn / sigma)

        # Only apply consistency where both prev and next are valid
        if mask_prev is not None and mask_next is not None:
            both_valid = (mask_prev & mask_next).astype(np.float32)
            consistency = consistency * both_valid + (1.0 - both_valid)

        w_prev *= consistency
        w_next *= consistency

    # Base always exists
    w_base = np.full(base_f.shape, base_weight, dtype=np.float32)

    weight_sum = w_base + w_prev + w_next + 1e-6

    out = (
        w_base * base_f +
        w_prev * prev_f +
        w_next * next_f
    ) / weight_sum

    out = np.clip(np.rint(out), 0, max_val).astype(np.uint16)

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

def fuse_chroma(base, prev, next_, mask_prev, mask_next, bit_depth=10, base_weight=0.5, sigma=30.0):
    base_f = base.astype(np.float32)
    prev_f = prev.astype(np.float32)
    next_f = next_.astype(np.float32)

    out = 0.6 * base_f
    weight = np.full(base.shape, 0.6, dtype=np.float32)

    out[mask_prev] += 0.2 * prev_f[mask_prev]
    weight[mask_prev] += 0.2

    out[mask_next] += 0.2 * next_f[mask_next]
    weight[mask_next] += 0.2

    out = out / weight
    return np.clip(np.rint(out), 0, 1023).astype(np.uint16)

def generate_frame(sr_model,b_y, b_u, b_v
            ,e_prev_y, e_prev_u, e_prev_v
            ,e_next_y, e_next_u, e_next_v):
        
        #upsample B[t] to 4K, downsample E[t-1] and E[t+1] to FHD for better flow estimation
        b_4k_y, b_4k_u, b_4k_v = upscale_yuv420_10bit_with_sr(sr_model, b_y, b_u, b_v)
        
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
        
        flow_prev_uv = resize_flow(flow_curr2prev, 1920, 1080)
        flow_next_uv = resize_flow(flow_curr2next, 1920, 1080)

        warped_prev_y, mask_prev_y = backward_warp_plane_with_mask(
            e_prev_y,
            flow_curr2prev_4k
        )

        warped_next_y, mask_next_y = backward_warp_plane_with_mask(
            e_next_y,
            flow_curr2next_4k
        )

        warped_prev_u, mask_prev_u = backward_warp_plane_with_mask(
            e_prev_u,
            flow_prev_uv
        )

        warped_prev_v, mask_prev_v = backward_warp_plane_with_mask(
            e_prev_v,
            flow_prev_uv
        )

        warped_next_u, mask_next_u = backward_warp_plane_with_mask(
            e_next_u,
            flow_next_uv
        )

        warped_next_v, mask_next_v = backward_warp_plane_with_mask(
            e_next_v,
            flow_next_uv
        )

        P_y = fuse_sources_with_mask_adaptive(
            b_4k_y,
            warped_prev_y,
            warped_next_y,
            mask_prev_y,
            mask_next_y,
            bit_depth=10,
            base_weight=0.5,
            sigma=30.0,
        )

        P_u = fuse_chroma(b_4k_u, warped_prev_u, warped_next_u, mask_prev_u, mask_next_u,bit_depth=10,
            base_weight=0.5,
            sigma=30.0,)
        P_v = fuse_chroma(b_4k_v, warped_prev_v, warped_next_v, mask_prev_v, mask_next_v,bit_depth=10,
            base_weight=0.5,
            sigma=30.0,)
        
        return P_y, P_u, P_v


