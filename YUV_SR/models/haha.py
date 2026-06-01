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
    ):
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
            window_size = 2,
            num_resblocks=num_resblocks
        )

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

    def forward(self, base, warped_hr):
        # Upsample the base layer using EDSR
        upsampled_base = self.edsr_layer(base)
        output = self.fusion_layer(upsampled_base, warped_hr)

        return output