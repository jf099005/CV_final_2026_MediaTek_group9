import argparse
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.yuv_dataset import (
    YOnlySRDataset,
    # YOnlySRValidationDataset,
)
from models.edsr import EDSREnd2EndModel


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
    parser.add_argument("--lr_end_factor", type=float, default=0.1,
                        help="LR decays linearly to lr * lr_end_factor by the final epoch")

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
        for (lr_y, prv_lr_y, nxt_lr_y), hr_y in val_loader:
            lr_y = lr_y.to(device)
            hr_y = hr_y.to(device)
            prv_lr_y = prv_lr_y.to(device)
            nxt_lr_y = nxt_lr_y.to(device)

            pred_y = model(lr_y, prv_lr_y, nxt_lr_y)
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

    train_dataset = YOnlySRDataset(
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
        dataset_type="train"
    )

    val_dataset = YOnlySRDataset(
        list_path=args.video_list,
        lr_width=args.lr_width,
        lr_height=args.lr_height,
        hr_width=args.hr_width,
        hr_height=args.hr_height,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        # stride=args.val_stride,
        samples_per_epoch=100,
        bit_depth=args.bit_depth,
        train_ratio=args.train_ratio,
        dataset_type="val"
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

    model = EDSREnd2EndModel(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=8,
        scale=args.scale,
    ).to(device)

    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

        for (lr_y, prv_lr_y, nx_lr_y), hr_y in pbar:
            lr_y = lr_y.to(device)
            hr_y = hr_y.to(device)
            prv_lr_y = prv_lr_y.to(device)
            nx_lr_y = nx_lr_y.to(device)

            pred_y = model(lr_y, prv_lr_y, nx_lr_y)
            loss = criterion(pred_y, hr_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = lr_y.size(0)
            total_loss += loss.item()# * batch_size
            total_samples += batch_size

            pbar.set_postfix(train_loss=loss.item())


        train_loss = total_loss / total_samples

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs}")
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