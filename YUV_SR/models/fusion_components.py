import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.pool(x))
        return x * weight


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class FusionLayer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=32,
        window_size=3,
        num_resblocks=4,
    ):
        super().__init__()

        # assert window_size == 3, "Currently supports prv/current/nxt"

        # -------------------------------------------------
        # Per-frame feature extraction
        # -------------------------------------------------
        self.feature_extractor = FeatureExtractor(
            in_channels,
            num_features
        )

        # -------------------------------------------------
        # Fusion head
        # concat(prv, cur, nxt)
        # channels = num_features * 3
        # -------------------------------------------------
        # self.fusion_head = nn.Conv2d(
        #     num_features * window_size,
        #     num_features,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )
        self.fusion_head = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1
        )


        # -------------------------------------------------
        # Residual fusion body
        # -------------------------------------------------
        body = []

        for _ in range(num_resblocks):
            body.append(ResBlock(num_features))

        body.append(SEBlock(num_features))

        self.fusion_body = nn.Sequential(*body)

        # -------------------------------------------------
        # Residual prediction
        # -------------------------------------------------
        self.fusion_tail = nn.Conv2d(
            num_features,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, upsampled_base, hr):

        # ---------------------------------------------
        # Feature extraction
        # ---------------------------------------------
        f_hr = self.feature_extractor(hr)
        f_cur = self.feature_extractor(upsampled_base)
        # f_nxt = self.feature_extractor(nxt)

        # ---------------------------------------------
        # Temporal fusion
        # ---------------------------------------------
        # fused = torch.cat(
        #     [f_prv, f_cur, f_nxt],
        #     dim=1
        # )

        fused = f_hr + f_cur

        feat = self.fusion_head(fused)

        # ---------------------------------------------
        # Residual processing
        # ---------------------------------------------
        feat = self.fusion_body(feat)

        # ---------------------------------------------
        # Predict residual
        # ---------------------------------------------
        residual = self.fusion_tail(feat)

        # ---------------------------------------------
        # Global residual learning
        # ---------------------------------------------
        output = upsampled_base + residual

        return output