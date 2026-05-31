import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_components import FusionLayer


# ---------------------------------------------------------------------------
# Inlined from CARN-pytorch/carn/model/ops.py
# ---------------------------------------------------------------------------

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()
        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])
        for p in self.shifter.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.shifter(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return F.relu(self.body(x) + x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                    nn.ReLU(inplace=True),
                    nn.PixelShuffle(2),
                ]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(3),
            ]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()
        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


# ---------------------------------------------------------------------------
# CARN-M block (weight-shared EResidualBlock, from carn_m.py)
# ---------------------------------------------------------------------------

class CARNMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(CARNMBlock, self).__init__()
        self.b1 = EResidualBlock(64, 64, group=group)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


# ---------------------------------------------------------------------------
# CARN-M backbone — wraps Net from carn_m.py with a single-arg forward
# ---------------------------------------------------------------------------

class CARNMBackbone(nn.Module):
    """
    CARN-M backbone initialised with multi_scale=True to match the pretrained
    checkpoint, which stores up2/up3/up4 heads.  forward() always uses scale=2.
    """

    def __init__(self, scale=2, group=4):
        super(CARNMBackbone, self).__init__()
        self.scale = scale

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = CARNMBlock(64, 64, group=group)
        self.b2 = CARNMBlock(64, 64, group=group)
        self.b3 = CARNMBlock(64, 64, group=group)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

        # multi_scale=True matches the pretrained checkpoint key layout
        self.upsample = UpsampleBlock(64, scale=scale, multi_scale=True, group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)
        out = self.exit(out)
        out = self.add_mean(out)
        return out

    def load_pretrained(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=True)
        if missing:
            print("[CARNMBackbone] Missing keys:", missing)
        if unexpected:
            print("[CARNMBackbone] Unexpected keys:", unexpected)
        else:
            print(f"[CARNMBackbone] Loaded pretrained weights: {checkpoint_path}")


# ---------------------------------------------------------------------------
# End-to-end model — same structure as EDSREnd2EndModel
# ---------------------------------------------------------------------------

class CARNEnd2EndModel(nn.Module):
    def __init__(
        self,
        out_channels=3,
        num_features=32,
        num_resblocks=4,
        scale=2,
        group=4,
        pretrained_path=None,
    ):
        super(CARNEnd2EndModel, self).__init__()
        self.carn_layer = CARNMBackbone(scale=scale, group=group)
        self.fusion_layer = FusionLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            num_features=num_features,
            window_size=3,
            num_resblocks=num_resblocks,
        )
        if pretrained_path is not None:
            self.carn_layer.load_pretrained(pretrained_path)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        m = cfg["model"]
        return cls(
            out_channels=m.get("out_channels", 3),
            num_features=m.get("num_features", 32),
            num_resblocks=m.get("fusion_num_blocks", 4),
            scale=m.get("scale", 2),
            group=m.get("group", 4),
            pretrained_path=m.get("carn_pretrained_path", None),
        )

    def forward(self, base, prv, nxt):
        upsampled_base = self.carn_layer(base)
        output = self.fusion_layer(upsampled_base, prv, nxt)
        return output
