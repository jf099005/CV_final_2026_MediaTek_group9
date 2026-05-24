import torch
import torch.nn as nn


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
    ):
        super().__init__()

        assert scale == 2, "This simple version currently supports scale x2 only."

        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = UpsamplerX2(num_features)

        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)

        residual = x
        x = self.body(x)
        x = self.body_tail(x)
        x = x + residual

        x = self.upsample(x)
        x = self.tail(x)

        return x