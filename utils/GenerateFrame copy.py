from utils.Resize import resize_yuv420_10bit
from utils.Wraping import backward_warp_plane_with_mask
from utils.RAFTFlow import RAFTFlowEstimator
from utils.YOnlySR import YOnlySR
import cv2
import numpy as np

raft_estimator = RAFTFlowEstimator()

def load_model(model_path):
    sr_model = YOnlySR(
        model_path=model_path,
        scale=2,
        bit_depth=10,
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
    # base_fhd = None
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
    # if base_fhd is None:
    #     h, w = base.shape[:2]
    #     base_fhd = cv2.resize(base, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    # prev_fhd = cv2.resize(prev, (prev.shape[1] // 2, prev.shape[0] // 2), interpolation=cv2.INTER_AREA)
    # next_fhd = cv2.resize(next_, (next_.shape[1] // 2, next_.shape[0] // 2), interpolation=cv2.INTER_AREA)

    base_fhd = base
    prev_fhd = prev
    next_fhd = next_

    base_f = base.astype(np.float32)
    prev_f = prev.astype(np.float32)
    next_f = next_.astype(np.float32)

    max_val = (1 << bit_depth) - 1

    # Residual error to base
    err_prev = np.abs(prev_fhd - base_fhd)
    err_next = np.abs(next_fhd - base_fhd)

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

def fuse_chroma(base, prev, next_, mask_prev, mask_next):
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
        
        e_prev_fhd_y, e_prev_fhd_u, e_prev_fhd_v = resize_yuv420_10bit(e_prev_y, e_prev_u, e_prev_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        e_next_fhd_y, e_next_fhd_u, e_next_fhd_v = resize_yuv420_10bit(e_next_y, e_next_u, e_next_v, 1920, 1080, interpolation=cv2.INTER_CUBIC)
        
        raft_w = 1280
        raft_h = 720

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

        # P_y = fuse_sources_with_mask_adaptive(
        #     b_4k_y,
        #     warped_prev_y,
        #     warped_next_y,
        #     mask_prev_y,
        #     mask_next_y,
        #     bit_depth=10,
        #     base_weight=0.3,
        #     sigma=30.0,
        #     base_fhd=b_y
        # )


        P_y, debug_msg = confidence_selection_fusion_y(
            b_y,
            b_4k_y,
            warped_prev_y,
            warped_next_y,
            mask_prev_y,
            mask_next_y,
            bit_depth=10,
            # base_weight=0.3,
            # sigma=30.0,
            # base_fhd=b_y
        )

        print('debug message:', debug_msg)

        P_u = fuse_chroma(b_4k_u, warped_prev_u, warped_next_u, mask_prev_u, mask_next_u)
        P_v = fuse_chroma(b_4k_v, warped_prev_v, warped_next_v, mask_prev_v, mask_next_v)
        
        return P_y, P_u, P_v, debug_msg




def confidence_selection_fusion_y(
    base_y_fhd,
    base_y_4k,
    warp_prev_y_4k,
    warp_next_y_4k,
    valid_prev_4k=None,
    valid_next_4k=None,
    bit_depth=10,
    blur_ksize=5,
    # thr_normal=50,
    # thr_edge=25,
):
    """
    Confidence-based fusion for Y channel.

    Inputs:
        base_y_fhd:      current base layer Y, shape = H x W
        base_y_4k:       SR / upscale result, shape = 2H x 2W
        warp_prev_y_4k:  previous 4K frame warped to current, shape = 2H x 2W
        warp_next_y_4k:  next 4K frame warped to current, shape = 2H x 2W
        valid_prev_4k:   valid mask for prev warp, shape = 2H x 2W, bool or 0/1
        valid_next_4k:   valid mask for next warp, shape = 2H x 2W, bool or 0/1

    Output:
        out_y: fused Y, shape = 2H x 2W
    """

    H, W = base_y_fhd.shape
    H4, W4 = base_y_4k.shape

    max_val = (1 << bit_depth) - 1

    # ------------------------------------------------------------
    # 1. Downsample warped 4K frames back to FHD
    # ------------------------------------------------------------
    prev_down = cv2.resize(
        warp_prev_y_4k,
        (W, H),
        interpolation=cv2.INTER_AREA
    )

    next_down = cv2.resize(
        warp_next_y_4k,
        (W, H),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------------
    # 2. Compute FHD difference against reliable base_y_fhd
    # ------------------------------------------------------------
    base_f = base_y_fhd.astype(np.float32)
    prev_f = prev_down.astype(np.float32)
    next_f = next_down.astype(np.float32)

    diff_prev = np.abs(prev_f - base_f)
    diff_next = np.abs(next_f - base_f)

    # ------------------------------------------------------------
    # 3. Local diff smoothing
    # ------------------------------------------------------------
    if blur_ksize > 1:
        diff_prev = cv2.blur(diff_prev, (blur_ksize, blur_ksize))
        diff_next = cv2.blur(diff_next, (blur_ksize, blur_ksize))

    # ------------------------------------------------------------
    # 4. Edge-aware threshold
    # ------------------------------------------------------------
    # Convert 10-bit / 12-bit Y to 8-bit for Canny
    # base_8bit = np.clip(base_f / max_val * 255.0, 0, 255).astype(np.uint8)

    # edge = cv2.Canny(base_8bit, 50, 150)
    # edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
    # edge = edge > 0

    thr_map = np.where(edge, thr_edge, thr_normal).astype(np.float32)

    # ------------------------------------------------------------
    # 5. Reliability mask at FHD
    # ------------------------------------------------------------
    prev_ok_fhd = diff_prev < thr_map
    next_ok_fhd = diff_next < thr_map

    # ------------------------------------------------------------
    # 6. Include valid masks if provided
    # ------------------------------------------------------------
    if valid_prev_4k is not None:
        valid_prev_fhd = cv2.resize(
            valid_prev_4k.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        prev_ok_fhd = prev_ok_fhd & valid_prev_fhd

    if valid_next_4k is not None:
        valid_next_fhd = cv2.resize(
            valid_next_4k.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        next_ok_fhd = next_ok_fhd & valid_next_fhd

    # ------------------------------------------------------------
    # 7. Prev / next competition
    # ------------------------------------------------------------
    # Only prev reliable -> use prev
    # Only next reliable -> use next
    # Both reliable -> choose smaller FHD diff
    # Neither reliable -> fallback to base_y_4k

    use_prev_fhd = prev_ok_fhd & (~next_ok_fhd | (diff_prev <= diff_next))
    use_next_fhd = next_ok_fhd & (~prev_ok_fhd | (diff_next < diff_prev))

    # ------------------------------------------------------------
    # 8. Upscale decision masks to 4K
    # ------------------------------------------------------------
    use_prev_4k = cv2.resize(
        use_prev_fhd.astype(np.uint8),
        (W4, H4),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    use_next_4k = cv2.resize(
        use_next_fhd.astype(np.uint8),
        (W4, H4),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # ------------------------------------------------------------
    # 9. Fusion
    # ------------------------------------------------------------
    out_y = base_y_4k.astype(np.float32).copy()

    out_y[use_prev_4k] = warp_prev_y_4k.astype(np.float32)[use_prev_4k]
    out_y[use_next_4k] = warp_next_y_4k.astype(np.float32)[use_next_4k]

    out_y = np.clip(out_y, 0, max_val).astype(np.uint16)

    debug = {
        "diff_prev_fhd": diff_prev,
        "diff_next_fhd": diff_next,
        "prev_ok_fhd": prev_ok_fhd,
        "next_ok_fhd": next_ok_fhd,
        "use_prev_fhd": use_prev_fhd,
        "use_next_fhd": use_next_fhd,
        # "edge_fhd": edge,
        "use_prev_4k": use_prev_4k,
        "use_next_4k": use_next_4k,
    }
    return out_y, debug


def save_mask_debug(debug, out_dir, frame_idx):
    import os
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_diff_prev.png"),
        np.clip(debug["diff_prev_fhd"] * 4, 0, 255).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_diff_next.png"),
        np.clip(debug["diff_next_fhd"] * 4, 0, 255).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_prev_ok.png"),
        debug["prev_ok_fhd"].astype(np.uint8) * 255
    )

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_next_ok.png"),
        debug["next_ok_fhd"].astype(np.uint8) * 255
    )

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_use_prev.png"),
        debug["use_prev_fhd"].astype(np.uint8) * 255
    )

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_use_next.png"),
        debug["use_next_fhd"].astype(np.uint8) * 255
    )

    cv2.imwrite(
        os.path.join(out_dir, f"{frame_idx:04d}_edge.png"),
        debug["edge_fhd"].astype(np.uint8) * 255
    )
