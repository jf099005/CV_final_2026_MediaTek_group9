import torch
import numpy as np
import cv2

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class RAFTFlowEstimator:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()

        self.model = raft_small(weights=weights, progress=True).to(self.device)
        self.model.eval()

    def y10_to_rgb_uint8(self, y):
        """
        RAFT expects RGB-like 3-channel images.
        Your input is 10-bit Y plane, so we duplicate Y into 3 channels.
        """
        y = np.clip(y, 0, 1023)
        y8 = (y / 4).astype(np.uint8)
        rgb = np.stack([y8, y8, y8], axis=-1)
        return rgb

    def compute_flow(self, img1_y, img2_y):
        """
        Compute optical flow from img1_y to img2_y.

        Input:
            img1_y: H x W, 10-bit Y plane
            img2_y: H x W, 10-bit Y plane

        Output:
            flow: H x W x 2, float32
                  flow[..., 0] = dx
                  flow[..., 1] = dy
        """

        img1 = self.y10_to_rgb_uint8(img1_y)
        img2 = self.y10_to_rgb_uint8(img2_y)

        # HWC uint8 -> CHW tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)

        # Add batch dimension
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        # Apply official RAFT transforms
        img1, img2 = self.transforms(img1, img2)

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        with torch.no_grad():
            # RAFT returns a list of flow predictions.
            # The last one is the final refined flow.
            flow_list = self.model(img1, img2)
            flow = flow_list[-1]

        # flow: 1 x 2 x H x W -> H x W x 2
        flow = flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        return flow