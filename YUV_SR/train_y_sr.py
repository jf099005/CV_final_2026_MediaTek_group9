import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset.png_dataset import PNGSRDataset
from dataset.yuv_dataset import YOnlySRDataset
from models.edsr import EDSREnd2EndModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr_yuv", required=True, help="Path to LR FHD YUV420 10-bit file")
    parser.add_argument("--hr_yuv", required=True, help="Path to HR 4K YUV420 10-bit file")

    parser.add_argument("--lr_width", type=int, default=1920)
    parser.add_argument("--lr_height", type=int, default=1080)
    parser.add_argument("--hr_width", type=int, default=3840)
    parser.add_argument("--hr_height", type=int, default=2160)

    parser.add_argument("--num_frames", type=int, required=True)

    parser.add_argument("--val_lr_yuv", default=None, help="Path to validation LR YUV file")
    parser.add_argument("--val_hr_yuv", default=None, help="Path to validation HR YUV file")
    parser.add_argument("--val_num_frames", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=1000)

    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--samples_per_epoch", type=int, default=10000)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--hr_png_dir", default=None, help="Optional directory of HR PNG images for additional training data")

    parser.add_argument("--save_dir", default="checkpoints_y")
    parser.add_argument("--resume", default=None)

    return parser.parse_args()


def save_curves(save_dir, train_history, val_history):
    csv_path = os.path.join(save_dir, "loss_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss"] + (["val_loss"] if val_history else [])
        writer.writerow(header)
        for i, t_loss in enumerate(train_history, 1):
            row = [i, t_loss]
            if val_history:
                row.append(val_history[i - 1] if i <= len(val_history) else "")
            writer.writerow(row)

    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.figure()
    epochs = range(1, len(train_history) + 1)
    plt.plot(epochs, train_history, label="train")
    if val_history:
        plt.plot(range(1, len(val_history) + 1), val_history, label="val")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Avg L1 Loss")
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def train():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    yuv_dataset = YOnlySRDataset(
        lr_yuv_path=args.lr_yuv,
        hr_yuv_path=args.hr_yuv,
        lr_width=args.lr_width,
        lr_height=args.lr_height,
        hr_width=args.hr_width,
        hr_height=args.hr_height,
        num_frames=args.num_frames,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
    )

    if args.hr_png_dir is not None:
        png_dataset = PNGSRDataset(
            hr_png_dir=args.hr_png_dir,
            scale=args.scale,
            lr_patch_size=args.patch_size,
            samples_per_epoch=args.samples_per_epoch,
        )
        dataset = ConcatDataset([yuv_dataset, png_dataset])
        print(f"Combined dataset: {len(yuv_dataset)} YUV + {len(png_dataset)} PNG samples")
    else:
        dataset = yuv_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = None
    if args.val_lr_yuv and args.val_hr_yuv:
        val_num_frames = args.val_num_frames or args.num_frames
        val_dataset = YOnlySRDataset(
            lr_yuv_path=args.val_lr_yuv,
            hr_yuv_path=args.val_hr_yuv,
            lr_width=args.lr_width,
            lr_height=args.lr_height,
            hr_width=args.hr_width,
            hr_height=args.hr_height,
            num_frames=val_num_frames,
            scale=args.scale,
            lr_patch_size=args.patch_size,
            samples_per_epoch=args.val_samples,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        print(f"Validation set: {len(val_dataset)} samples")

    # model = EDSRSmall(
    #     in_channels=1,
    #     out_channels=1,
    #     num_features=64,
    #     num_blocks=8,
    #     scale=args.scale,
    # ).to(device)

    model = EDSREnd2EndModel(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=16,
        scale=args.scale,
    ).to(device)

    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float("inf")
    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
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

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(dataloader)
        train_history.append(avg_train_loss)

        log_msg = f"Epoch {epoch}: train_loss = {avg_train_loss:.6f}"

        monitor_loss = avg_train_loss

        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for (lr_y, prv_lr_y, nx_lr_y), hr_y in tqdm(val_dataloader, desc="  Val", leave=False):
                    lr_y = lr_y.to(device)
                    prv_lr_y = prv_lr_y.to(device)
                    nx_lr_y = nx_lr_y.to(device)
                    hr_y = hr_y.to(device)
                    pred_y = model(lr_y, prv_lr_y, nx_lr_y)
                    total_val_loss += criterion(pred_y, hr_y).item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            val_history.append(avg_val_loss)
            log_msg += f"  val_loss = {avg_val_loss:.6f}"
            monitor_loss = avg_val_loss

        scheduler.step()
        print(log_msg + f"  lr = {scheduler.get_last_lr()[0]:.2e}")

        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model: {best_path}")

        save_curves(args.save_dir, train_history, val_history)


if __name__ == "__main__":
    train()
