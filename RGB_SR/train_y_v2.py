import argparse
import json
import os
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.png_dataset import PNGSRDataset
from models.carn_v1 import CARNEnd2EndModel

PROJECT_ROOT = '/mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9'
sys.path.append(PROJECT_ROOT)


import pytorch_msssim
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)


class CombinedLoss(nn.Module):
    """
    Loss = w_ssim * (1-SSIM) + w_l1 * L1 + w_grad * Gradient + w_fft * FFT
    Gradient loss penalises missing edges; FFT loss penalises missing high-freq texture.
    """
    def __init__(self, w_ssim=0.70, w_l1=0.15, w_grad=0.10, w_fft=0.05):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_l1   = w_l1
        self.w_grad = w_grad
        self.w_fft  = w_fft
        self.l1 = nn.L1Loss()

    def _gradient_loss(self, pred, target):
        # Build Sobel kernels on the same device/dtype as pred every call.
        # Cheap: just a view + to(), no allocation when already correct.
        kx = _SOBEL_X.expand(3, 1, 3, 3).contiguous().to(device=pred.device, dtype=pred.dtype)
        ky = _SOBEL_Y.expand(3, 1, 3, 3).contiguous().to(device=pred.device, dtype=pred.dtype)
        gx_p = F.conv2d(pred,   kx, groups=3, padding=1)
        gx_t = F.conv2d(target, kx, groups=3, padding=1)
        gy_p = F.conv2d(pred,   ky, groups=3, padding=1)
        gy_t = F.conv2d(target, ky, groups=3, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

    def _fft_loss(self, pred, target):
        p = torch.fft.rfft2(pred.float(),   norm="ortho")
        t = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(p.abs(), t.abs())

    def forward(self, pred, target):
        ssim_loss = 1 - pytorch_msssim.ssim(pred, target, data_range=1.0)
        l1_loss   = self.l1(pred, target)
        grad_loss = self._gradient_loss(pred, target)
        fft_loss  = self._fft_loss(pred, target)
        return (
            self.w_ssim * ssim_loss
            + self.w_l1  * l1_loss
            + self.w_grad * grad_loss
            + self.w_fft  * fft_loss
        )

def plot_loss_curve(log_path, save_dir):
    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss")
    ax.plot(epochs, val_losses, marker="s", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_list", required=True,
                        help="CSV listing (lr_png_dir, even_hr_png_dir, hr_png_dir, num_frames) per row")

    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--samples_per_epoch", type=int, default=10000)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_end_factor", type=float, default=0.01,
                        help="LR decays linearly to lr * lr_end_factor by the final epoch")

    parser.add_argument("--save_dir", default="checkpoints_rgb")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze CARNMBackbone weights and only train fusion_layer")

    return parser.parse_args()


def freeze_backbone(model):
    for param in model.carn_layer.parameters():
        param.requires_grad = False
    print("[freeze_backbone] CARNMBackbone parameters frozen")


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (lr_rgb, prv_rgb, nxt_rgb), hr_rgb in val_loader:
            lr_rgb = lr_rgb.to(device)
            prv_rgb = prv_rgb.to(device)
            nxt_rgb = nxt_rgb.to(device)
            hr_rgb = hr_rgb.to(device)

            pred = model(lr_rgb, prv_rgb, nxt_rgb)
            loss = criterion(pred, hr_rgb)

            batch_size = lr_rgb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, "loss_log.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dataset = PNGSRDataset(
        list_path=args.train_list,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        train_ratio=args.train_ratio,
        dataset_type="train",
    )

    val_dataset = PNGSRDataset(
        list_path=args.train_list,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        samples_per_epoch=int(args.samples_per_epoch*(1-args.train_ratio)),
        train_ratio=args.train_ratio,
        dataset_type="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Train list:", args.train_list)
    print("Train ratio:", args.train_ratio)
    print("Train samples per epoch:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    config_path = os.path.join(os.path.dirname(__file__), "models", "model_config.json")
    model = CARNEnd2EndModel.from_config(config_path).to(device)

    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))

    if args.freeze_backbone:
        freeze_backbone(model)

    # criterion = nn.L1Loss()/
    # criterion = nn.MSELoss()
    criterion = CombinedLoss().to(device)
    trainable_params = (
        model.fusion_layer.parameters()
        if args.freeze_backbone
        else model.parameters()
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.lr_end_factor,
        total_iters=args.epochs,
    )

    

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for (lr_rgb, prv_rgb, nxt_rgb), hr_rgb in pbar:
            lr_rgb = lr_rgb.to(device)
            prv_rgb = prv_rgb.to(device)
            nxt_rgb = nxt_rgb.to(device)
            hr_rgb = hr_rgb.to(device)

            pred = model(lr_rgb, prv_rgb, nxt_rgb)
            loss = criterion(pred, hr_rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = lr_rgb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix(train_loss=loss.item())

        train_loss = total_loss / total_samples

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]")
        val_loss = validate(model, val_pbar, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}: "
            f"train_loss = {train_loss:.6f}, "
            f"val_loss = {val_loss:.6f}, "
            f"lr = {current_lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        plot_loss_curve(log_path, args.save_dir)

        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")


if __name__ == "__main__":
    train()
