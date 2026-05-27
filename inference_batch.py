import argparse
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from YUV_SR.dataset.yuv_dataset import (
    MultiVideoYOnlySRDataset,
    MultiVideoYOnlySRValidationDataset,
)
from YUV_SR.models.edsr_small import EDSRSmall


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate missing 4K odd frames using FHD base layer and adjacent 4K enhancement frames"
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Path to FHD base-layer YUV file"
    )

    parser.add_argument(
        "--enhancement",
        required=True,
        help="Path to 4K enhancement-layer YUV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output generated 4K YUV file"
    )

    parser.add_argument(
        "--base_width",
        type=int,
        default=1920,
        help="Base-layer width"
    )

    parser.add_argument(
        "--base_height",
        type=int,
        default=1080,
        help="Base-layer height"
    )

    parser.add_argument(
        "--enh_width",
        type=int,
        default=3840,
        help="Enhancement-layer width"
    )

    parser.add_argument(
        "--enh_height",
        type=int,
        default=2160,
        help="Enhancement-layer height"
    )

    return parser.parse_args()


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
    print("Training 3-channel YUV super-resolution model (2x upsampling)")
    print(f"Input: {args.lr_width}x{args.lr_height} -> Output: {args.hr_width}x{args.hr_height}")

    train_dataset = MultiVideoYOnlySRDataset(
        list_path=args.video_list,
        lr_width=args.lr_width,
        lr_height=args.lr_height,
        hr_width=args.hr_width,
        hr_height=args.hr_height,
        scale=2,
        lr_patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        bit_depth=args.bit_depth,
        train_ratio=args.train_ratio,
        channels=3,  # 3-channel YUV
    )

    val_dataset = MultiVideoYOnlySRValidationDataset(
        list_path=args.video_list,
        lr_width=args.lr_width,
        lr_height=args.lr_height,
        hr_width=args.hr_width,
        hr_height=args.hr_height,
        scale=2,
        lr_patch_size=args.patch_size,
        stride=args.val_stride,
        bit_depth=args.bit_depth,
        train_ratio=args.train_ratio,
        channels=3,  # 3-channel YUV
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

    print("Video list:", args.video_list)
    print("Train ratio:", args.train_ratio)
    print("Train samples per epoch:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    model = EDSRSmall(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=8,
        scale=2,
    ).to(device)

    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr_yuv, hr_yuv in pbar:
            lr_yuv = lr_yuv.to(device)
            hr_yuv = hr_yuv.to(device)

            pred_yuv = model(lr_yuv)
            loss = criterion(pred_yuv, hr_yuv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = lr_yuv.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix(train_loss=loss.item())

        train_loss = total_loss / total_samples
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss = {train_loss:.6f}, "
            f"val_loss = {val_loss:.6f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")


if __name__ == "__main__":
    train()
