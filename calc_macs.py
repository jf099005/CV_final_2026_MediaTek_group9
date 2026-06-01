"""
Compute MACs/pixel for the entire inference pipeline.

Pipeline per odd frame:
  1. EDSRSmall x2 SR    (FHD Y 1920x1080 -> 4K Y 3840x2160)
  2. RAFT-Small x2      (flow at 1280x720, twice)
  3. Backward warp x6   (cv2.remap, traditional)
  4. Adaptive fusion x3 (element-wise, traditional)

Output resolution: 3840x2160 = 8,294,400 pixels
"""
import sys, os
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "YUV_SR"))
sys.path.insert(0, PROJECT_ROOT)

from YUV_SR.models.edsr_small import EDSRSmall
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from ptflops import get_model_complexity_info

# ── Resolution constants ──────────────────────────────────────────────────────
FHD_H,  FHD_W  = 1080, 1920
UHD_H,  UHD_W  = 2160, 3840
RAFT_H, RAFT_W = 720,  1280
CHR_H,  CHR_W  = FHD_H // 2, FHD_W // 2   # 4K chroma = FHD after SR

OUTPUT_PIXELS = UHD_H * UHD_W   # denominator for MACs/pixel
print(f"Output frame: {UHD_W}×{UHD_H} = {OUTPUT_PIXELS:,} pixels\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. EDSRSmall  (input: 1×1080×1920 Y-only)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. EDSRSmall x2 SR  (FHD Y → 4K Y)")
print("=" * 60)

sr_model = EDSRSmall(in_channels=1, out_channels=1,
                     num_features=48, num_blocks=8, scale=2)  # num_features: 64 -> 48
sr_model.eval()

macs_sr, params_sr = get_model_complexity_info(
    sr_model,
    (1, FHD_H, FHD_W),
    as_strings=False,
    print_per_layer_stat=True,
    verbose=False,
)
print(f"\nEDSRSmall  MACs : {macs_sr/1e9:.3f} GMACs")
print(f"EDSRSmall  Params: {params_sr/1e6:.3f} M\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RAFT-Small  (input: two 3×720×1280 images, called twice)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. RAFT-Small optical flow  (1280×720, ×2 calls)")
print("=" * 60)

# RAFT takes two images; wrap it so ptflops sees a single (6, H, W) tensor
class RAFTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (B, 6, H, W) – first 3 ch = img1, last 3 = img2
        img1, img2 = x[:, :3], x[:, 3:]
        return self.model(img1, img2)

weights = Raft_Small_Weights.DEFAULT
raft_model = raft_small(weights=weights)
raft_model.eval()
raft_wrapper = RAFTWrapper(raft_model)

macs_raft, params_raft = get_model_complexity_info(
    raft_wrapper,
    (6, RAFT_H, RAFT_W),
    as_strings=False,
    print_per_layer_stat=True,
    verbose=False,
)
print(f"\nRAFT-Small MACs (1 call): {macs_raft/1e9:.3f} GMACs")
print(f"RAFT-Small MACs (×2 calls): {2*macs_raft/1e9:.3f} GMACs")
print(f"RAFT-Small Params: {params_raft/1e6:.3f} M\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Backward warp (cv2.remap = bilinear interpolation)
#    Each remap pixel needs: 4 multiplies + 3 adds ≈ 7 MACs  (bilinear interp)
#    Planes warped:
#      prev_y, next_y  at 3840×2160
#      prev_u, next_u  at 1920×1080
#      prev_v, next_v  at 1920×1080
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("3. Backward warp (cv2.remap, bilinear)")
print("=" * 60)

BILINEAR_MACS_PER_PIXEL = 7   # 4 mul + 3 add for bilinear interpolation

warp_y_pixels    = UHD_H * UHD_W          # 3840×2160
warp_chroma_pixels = FHD_H * FHD_W        # 1920×1080

macs_warp = (
    2 * warp_y_pixels * BILINEAR_MACS_PER_PIXEL       # prev_y + next_y
  + 4 * warp_chroma_pixels * BILINEAR_MACS_PER_PIXEL  # u/v × prev/next
)
print(f"  Y planes  (3840×2160 × 2): {2*warp_y_pixels*BILINEAR_MACS_PER_PIXEL/1e9:.4f} GMACs")
print(f"  UV planes (1920×1080 × 4): {4*warp_chroma_pixels*BILINEAR_MACS_PER_PIXEL/1e9:.4f} GMACs")
print(f"  Warp total               : {macs_warp/1e9:.4f} GMACs\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Adaptive fusion  (fuse_sources_with_mask_adaptive)
#    Per pixel ops:
#      err_prev  = |prev - base|               → 1 sub + 1 abs  ≈ 2
#      err_next  = |next - base|               → 2
#      w_prev    = exp(-err / sigma)            → 1 div + 1 exp + 1 mul ≈ 3
#      w_next    → 3
#      mask_apply× 2                           → 2 mul
#      err_pn    = |prev - next|               → 2
#      consistency = exp(-err_pn / sigma)      → 3
#      w_prev   *= consistency                 → 1
#      w_next   *= consistency                 → 1
#      weight_sum = w_base + w_prev + w_next   → 2 add
#      numerator  = w_base*base + w_prev*prev + w_next*next  → 3 mul + 2 add
#      out       = numerator / weight_sum      → 1 div
#      clip/round                              → ~2
#    Total ≈ 27 MACs/pixel  (conservative estimate)
#
#    Planes fused: Y at 3840×2160, U and V each at 1920×1080
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("4. Adaptive fusion (fuse_sources_with_mask_adaptive)")
print("=" * 60)

FUSION_MACS_PER_PIXEL = 27

macs_fusion_y  = UHD_H * UHD_W * FUSION_MACS_PER_PIXEL
macs_fusion_uv = 2 * FHD_H * FHD_W * FUSION_MACS_PER_PIXEL
macs_fusion = macs_fusion_y + macs_fusion_uv
print(f"  Y  (3840×2160 × 27 ops): {macs_fusion_y/1e9:.4f} GMACs")
print(f"  UV (1920×1080 × 2 × 27 ops): {macs_fusion_uv/1e9:.4f} GMACs")
print(f"  Fusion total: {macs_fusion/1e9:.4f} GMACs\n")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY  (per odd frame)")
print("=" * 60)

total_macs = macs_sr + 2 * macs_raft + macs_warp + macs_fusion

rows = [
    ("EDSRSmall SR  (×1)",   macs_sr),
    ("RAFT-Small    (×2)",   2 * macs_raft),
    ("Backward warp (×6)",   macs_warp),
    ("Adaptive fuse (×3)",   macs_fusion),
    ("TOTAL",                total_macs),
]
col_w = 26
print(f"{'Component':<{col_w}} {'GMACs':>10}  {'MACs/pixel':>12}")
print("-" * (col_w + 25))
for name, m in rows:
    print(f"{name:<{col_w}} {m/1e9:>10.3f}  {m/OUTPUT_PIXELS:>12.3f}")

print()
print(f"Output pixels            : {OUTPUT_PIXELS:,}  (3840×2160)")
print(f"Total MACs/output-pixel  : {total_macs/OUTPUT_PIXELS:.3f}")
print(f"  Neural only (SR+RAFT)  : {(macs_sr + 2*macs_raft)/OUTPUT_PIXELS:.3f} MACs/pixel")
print(f"  Traditional (warp+fuse): {(macs_warp + macs_fusion)/OUTPUT_PIXELS:.6f} MACs/pixel")
