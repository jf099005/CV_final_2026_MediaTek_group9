
import sys
PROJECT_ROOT = '/mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9'
sys.path.append(PROJECT_ROOT)

import cv2
from utils.Resize import resize_yuv420_10bit
from utils.GenerateFrame import resize_flow, backward_warp_plane_with_mask

def warping_hr(
            raft_estimator,
            b_y, b_u, b_v,
            e_prev_y, e_prev_u, e_prev_v,
            e_next_y, e_next_u, e_next_v,
        ):

        raft_w = e_prev_y.shape[1]
        raft_h = e_prev_y.shape[0]

        b_raft_y, _, _ = resize_yuv420_10bit(
            b_y, b_u, b_v,
            raft_w, raft_h,
            interpolation=cv2.INTER_AREA
        )

        # e_prev_raft_y, _, _ = resize_yuv420_10bit(
        #     e_prev_y, e_prev_u, e_prev_v,
        #         raft_w, raft_h,
        #         interpolation=cv2.INTER_AREA
        #     )

        # e_next_raft_y, _, _ = resize_yuv420_10bit(
        #     e_next_y, e_next_u, e_next_v,
        #     raft_w, raft_h,
        #     interpolation=cv2.INTER_AREA
        #     )

        flow_curr2prev = raft_estimator.compute_flow(
            # b_raft_y,
            b_raft_y,
            # e_prev_raft_y
            e_prev_y
            )

        flow_curr2next = raft_estimator.compute_flow(
            # b_raft_y,
            b_raft_y,
            e_next_y
            # e_next_raft_y
            )

        # flow_curr2prev_4k = resize_flow(
        #     flow_curr2prev,
        #     out_width=3840,
        #     out_height=2160
        #     )

        # flow_curr2next_4k = resize_flow(
        #     flow_curr2next,
        #     out_width=3840,
        #     out_height=2160
        #     )

        warped_prev_y, mask_prev_y = backward_warp_plane_with_mask(
            e_prev_y,
            flow_curr2prev
        )

        warped_next_y, mask_next_y = backward_warp_plane_with_mask(
            e_next_y,
            flow_curr2next
        )

        # warped_prev_u, mask_prev_u = backward_warp_plane_with_mask(
        #     e_prev_u,
        #     flow_prev_uv
        # )

        # warped_prev_v, mask_prev_v = backward_warp_plane_with_mask(
        #     e_prev_v,
        #     flow_prev_uv
        # )

        # warped_next_u, mask_next_u = backward_warp_plane_with_mask(
        #     e_next_u,
        #     flow_next_uv
        # )

        # warped_next_v, mask_next_v = backward_warp_plane_with_mask(
        #     e_next_v,
        #     flow_next_uv
        # )

        return warped_prev_y, mask_prev_y, warped_next_y, mask_next_y #, warped_prev_u, mask_prev_u, warped_prev_v, mask_prev_v, warped_next_u, mask_next_u, warped_next_v, mask_next_v

        # P_y = fuse_sources_with_mask_adaptive(
        #     b_4k_y,
        #     warped_prev_y,
        #     warped_next_y,
        #     mask_prev_y,
        #     mask_next_y,
        #     bit_depth=10,
        #     base_weight=0.3,
        #     sigma=30.0,
        # )

        # P_u = fuse_chroma(b_4k_u, warped_prev_u, warped_next_u, mask_prev_u, mask_next_u)
        # P_v = fuse_chroma(b_4k_v, warped_prev_v, warped_next_v, mask_prev_v, mask_next_v)
        
        # return P_y, P_u, P_v

