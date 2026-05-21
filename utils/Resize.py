import cv2


def resize_yuv420_10bit(y, u, v, out_width, out_height, interpolation=cv2.INTER_NEAREST):
    """
    Resize a 10-bit YUV420 frame to any target resolution.

    Parameters
    ----------
    y : np.ndarray
        Y plane, shape = (height, width)

    u : np.ndarray
        U plane, shape = (height // 2, width // 2)

    v : np.ndarray
        V plane, shape = (height // 2, width // 2)

    out_width : int
        Target output width

    out_height : int
        Target output height

    interpolation : int
        OpenCV interpolation method.
        Example:
            cv2.INTER_NEAREST
            cv2.INTER_LINEAR
            cv2.INTER_CUBIC
            cv2.INTER_AREA

    Returns
    -------
    y_out, u_out, v_out : np.ndarray
        Resized YUV420 10-bit planes.
    """

    if out_width % 2 != 0 or out_height % 2 != 0:
        raise ValueError("YUV420 requires output width and height to be even.")

    y_out = cv2.resize(
        y,
        (out_width, out_height),
        interpolation=interpolation
    )

    u_out = cv2.resize(
        u,
        (out_width // 2, out_height // 2),
        interpolation=interpolation
    )

    v_out = cv2.resize(
        v,
        (out_width // 2, out_height // 2),
        interpolation=interpolation
    )

    return y_out, u_out, v_out