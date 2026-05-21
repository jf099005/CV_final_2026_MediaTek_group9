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

def resize_flow(flow, out_width, out_height):
    """
    Resize flow to another resolution, and scale displacement accordingly.
    """
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
def forward_warp_plane(plane, flow):
    """
    Forward warp a single 2D plane using flow.

    Input:
        plane: (H, W)
        flow:  (H, W, 2)
               flow[..., 0] = dx
               flow[..., 1] = dy

    Output:
        warped: (H, W)
        mask:   (H, W)  
    """
    h, w = plane.shape

    # source pixel coordinates
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # target positions
    x_new = np.rint(xs + flow[..., 0]).astype(np.int32)
    y_new = np.rint(ys + flow[..., 1]).astype(np.int32)

    valid = (
        (x_new >= 0) & (x_new < w) &
        (y_new >= 0) & (y_new < h)
    )

    warped_sum = np.zeros((h, w), dtype=np.float32)
    warped_cnt = np.zeros((h, w), dtype=np.float32)

    np.add.at(warped_sum, (y_new[valid], x_new[valid]), plane[valid].astype(np.float32))
    np.add.at(warped_cnt, (y_new[valid], x_new[valid]), 1.0)

    mask = warped_cnt > 0
    warped = np.zeros((h, w), dtype=np.float32)
    warped[mask] = warped_sum[mask] / warped_cnt[mask]

    if np.issubdtype(plane.dtype, np.integer):
        warped = np.rint(warped).astype(plane.dtype)
    else:
        warped = warped.astype(plane.dtype)

    return warped, mask


def forward_warp_yuv420(y, u, v, flow_y):
    """
    Warp a YUV420 frame into current-frame coordinates using 4K luma flow.

    Input:
        y:      (H, W)
        u:      (H/2, W/2)
        v:      (H/2, W/2)
        flow_y: (H, W, 2)  flow on luma resolution

    Output:
        warped_y, warped_u, warped_v
        mask_y, mask_u, mask_v
    """

    # warp Y directly
    warped_y, mask_y = forward_warp_plane(y, flow_y)

    # resize flow for U/V
    uv_h, uv_w = u.shape
    flow_uv = resize_flow(flow_y, uv_w, uv_h)

    warped_u, mask_u = forward_warp_plane(u, flow_uv)
    warped_v, mask_v = forward_warp_plane(v, flow_uv)

    return warped_y, warped_u, warped_v, mask_y, mask_u, mask_v