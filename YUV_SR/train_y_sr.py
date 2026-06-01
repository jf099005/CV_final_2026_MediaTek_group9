import argparse
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.yuv_dataset import (
    MultiVideoYOnlySRDataset,
    MultiVideoYOnlySRValidationDataset,
)
from models.edsr_small import EDSRSmall


def parse_args():
    parser = argparse.ArgumentParser()

    # 多影片訓練：改成讀 video_list
    parser.add_argument("--video_list", required=True, help="Path to video list csv/txt")

    parser.add_argument("--lr_width", type=int, default=1920)
    parser.add_argument("--lr_height", type=int, default=1080)
    parser.add_argument("--hr_width", type=int, default=3840)
    parser.add_argument("--hr_height", type=int, default=2160)

    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--samples_per_epoch", type=int, default=10000)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--val_stride", type=int, default=384)
    parser.add_argument("--bit_depth", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--save_dir", default="checkpoints_y")
    parser.add_argument("--resume", default=None)

    return parser.parse_args()


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for lr_y, hr_y in val_loader:
            lr_y = lr_y.to(device)
            hr_y = hr_y.to(device)

            pred_y = model(lr_y)
            loss = criterion(pred_y, hr_y)

            batch_size = lr_y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


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

    train_dataset = MultiVideoYOnlySRDataset(
        list_path=args.video_list,
        lr_width=args.lr_width,
        lr_height=args.lr_height,
        hr_width=args.hr_width,
        hr_height=args.hr_height,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        bit_depth=args.bit_depth,
        train_ratio=args.train_ratio,
    )

    val_dataset = MultiVideoYOnlySRValidationDataset(
        list_path=args.video_list,
        lr_width=args.lr_width,
        lr_height=args.lr_height,
        hr_width=args.hr_width,
        hr_height=args.hr_height,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        stride=args.val_stride,
        bit_depth=args.bit_depth,
        train_ratio=args.train_ratio,
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
        in_channels=1,
        out_channels=1,
        num_features=48,
        num_blocks=8,
        scale=args.scale,
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

        for lr_y, hr_y in pbar:
            lr_y = lr_y.to(device)
            hr_y = hr_y.to(device)

            pred_y = model(lr_y)
            loss = criterion(pred_y, hr_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = lr_y.size(0)
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