from utils.YOnlySR import YOnlySR


def generate_frame(
    sr_model,
    b_y, b_u, b_v,
    e_prev_y, e_prev_u, e_prev_v,
    e_next_y, e_next_u, e_next_v,
):
    """
    Generate a 4K frame using EDSREnd2EndModel.

    Y channel: model takes LR base Y + HR prev/next Y frames end-to-end.
    U/V channels: bicubic resize from the base U/V planes.

    Args:
        sr_model:  YOnlySR instance wrapping EDSREnd2EndModel
        b_y/u/v:   FHD base-layer planes
        e_prev_y:  4K Y of the previous enhancement frame (used as 'prv' input)
        e_next_y:  4K Y of the next enhancement frame (used as 'nxt' input)
        e_prev_u/v, e_next_u/v: accepted for API compatibility but unused
    """
    P_y, P_u, P_v = sr_model.upscale_yuv420(b_y, b_u, b_v, e_prev_y, e_next_y)
    return P_y, P_u, P_v
