import cv2
import numpy as np

def backward_warp_plane_with_mask(plane, flow):
    h, w = plane.shape

    xs, ys = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = xs + flow[..., 0].astype(np.float32)
    map_y = ys + flow[..., 1].astype(np.float32)

    valid = (
        (map_x >= 0) & (map_x <= w - 1) &
        (map_y >= 0) & (map_y <= h - 1)
    )

    warped = cv2.remap(
        plane,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped, valid