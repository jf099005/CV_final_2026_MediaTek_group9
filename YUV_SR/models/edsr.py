import json
import torch
import torch.nn as nn
from .fusion_components import FusionLayer

class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class UpsamplerX2(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class EDSRSmall(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=8,
        scale=2,
        do_upsample=True
    ):
        super().__init__()

        assert scale == 2, "This simple version currently supports scale x2 only."

        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        # if do_upsample:
        self.upsample = UpsamplerX2(num_features)

        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)

        residual = x
        x = self.body(x)
        x = self.body_tail(x)
        x = x + residual

        # if self.upsample is not None:
        x = self.upsample(x)
            
        x = self.tail(x)

        return x
    
class EDSREnd2EndModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=32,
        num_blocks=8,
        num_resblocks=4,
        scale=2,
        window_size=3
    ):
        assert window_size == 3, "This model currently supports a temporal window size of 3 (previous, current, next)."
        super().__init__()
        self.edsr_layer = EDSRSmall(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            scale=scale
        )

        self.fusion_layer = FusionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            window_size=window_size,
            num_resblocks=num_resblocks
        )

        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(1, num_features, 3,1,1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(num_features, num_features, 3,1,1),
        # )

        # self.fusion_layer = nn.Sequential(
        #     nn.Conv2d(in_channels * window_size, num_features, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features, num_features, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features, out_channels, 3, 1, 1),
        # )

        # self.fusion_layer = nn.Sequential(
        #     nn.Conv2d(in_channels * window_size, num_features, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
        # )

    @classmethod
    def from_config(cls, config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        m = cfg["model"]
        return cls(
            in_channels=m["in_channels"],
            out_channels=m["out_channels"],
            num_features=m["num_features"],
            num_blocks=m["num_blocks"],
            num_resblocks=m.get("fusion_num_blocks", m["num_blocks"]),
            scale=m["scale"],
        )

    # def process_prv_frame(self, frame):
    #     # Process the previous frame (e.g., downsample or extract features)
    #     return frame

    def forward(self, base, prv, nxt):
        # Upsample the base layer using EDSR
        upsampled_base = self.edsr_layer(base)

        # Concatenate the upsampled base and enhancement layer
        # fused_input = torch.cat([prv, upsampled_base, nxt], dim=1)

        # Fuse the features to produce the final output
        # output = self.fusion_layer(fused_input)
        output = self.fusion_layer(upsampled_base, prv, nxt)

        return output



# class EDSREnd2EndModel(nn.Module):
#     def __init__(
#         self,
#         in_channels=1,
#         out_channels=1,
#         num_features=64,
#         num_blocks=8,
#         scale=2,
#         window_size=3
#     ):
#         assert window_size == 3, "This model currently supports a temporal window size of 3 (previous, current, next)."
#         super().__init__()

#         self.upsample = UpsamplerX2(1)

#         self.hr_prv_y_head = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride = 2),
#             # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         self.hr_nxt_y_head = nn.Sequential(
#             nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride = 2),
#             # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # self.shared_head = nn.Sequential(
#         #     nn.Conv2d(in_channels, num_features, 3, 1, 1),
#         #     nn.ReLU(inplace=True),
#         # )


#         self.fusion_layer = nn.Sequential(
#             nn.Conv2d(num_features * window_size, num_features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
#         )
        
#         # self.fusion_layer = nn.Sequential(
#         #     nn.Conv2d(192, 64, 3, 1, 1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(64, 64, 3, 1, 1),
#         # )


#         self.edsr_layer = EDSRSmall(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             num_features=num_features,
#             num_blocks=num_blocks,
#             scale=scale,
#             # do_upsample=False
#         )

#     def process_prv_frame(self, frame):
#         # Process the previous frame (e.g., downsample or extract features)
#         return frame


#     def forward(self, base, prv, nxt):
#         base_emb = self.edsr_layer.head(base)
#         prv_emb = self.hr_prv_y_head(prv)
#         nxt_emb = self.hr_nxt_y_head(nxt)

#         # Concatenate the upsampled base and enhancement layer
#         fused_input = torch.cat([prv_emb, base_emb, nxt_emb], dim=1)

#         # Fuse the features to produce the final output
#         edsr_emb = self.fusion_layer(fused_input)


#         #running EDSR

#         residual = edsr_emb
#         x = self.edsr_layer.body(edsr_emb)
#         x = self.edsr_layer.body_tail(x)
#         x = x + residual

#         x = self.edsr_layer.upsample(x)
#         output = self.edsr_layer.tail(x)

#         return output