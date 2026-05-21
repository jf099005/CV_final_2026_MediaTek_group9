import cv2

def upscale_yuv420_10bit(y, u, v, out_width, out_height):
    y_4k = cv2.resize(
        y,
        (out_width, out_height),
        interpolation=cv2.INTER_NEAREST
    )

    u_4k = cv2.resize(
        u,
        (out_width // 2, out_height // 2),
        interpolation=cv2.INTER_NEAREST
    )

    v_4k = cv2.resize(
        v,
        (out_width // 2, out_height // 2),
        interpolation=cv2.INTER_NEAREST
    )

    return y_4k, u_4k, v_4k