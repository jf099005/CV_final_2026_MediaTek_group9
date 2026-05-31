import argparse
import json
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.RLR_dataset import (
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
    _default_config = os.path.join(os.path.dirname(__file__), "models/model_config.json")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=_default_config)
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config) as f:
        cfg = json.load(f)

    d = cfg.get("data", {})
    t = cfg.get("training", {})

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pre_args.config, help="Path to model_config.json")

    # 多影片訓練：改成讀 video_list
    parser.add_argument("--video_list", required=True, help="Path to video list csv/txt")

    parser.add_argument("--lr_width",  type=int,   default=d.get("lr_width",  1920))
    parser.add_argument("--lr_height", type=int,   default=d.get("lr_height", 1080))
    parser.add_argument("--hr_width",  type=int,   default=d.get("hr_width",  3840))
    parser.add_argument("--hr_height", type=int,   default=d.get("hr_height", 2160))

    parser.add_argument("--patch_size",        type=int,   default=d.get("patch_size",        96))
    parser.add_argument("--samples_per_epoch", type=int,   default=d.get("samples_per_epoch", 10000))
    parser.add_argument("--val_stride",        type=int,   default=d.get("val_stride",        384))
    parser.add_argument("--bit_depth",         type=int,   default=d.get("bit_depth",         10))
    parser.add_argument("--train_ratio",       type=float, default=d.get("train_ratio",       0.8))

    parser.add_argument("--batch_size",    type=int,   default=t.get("batch_size",    8))
    parser.add_argument("--epochs",        type=int,   default=t.get("epochs",        50))
    parser.add_argument("--lr",            type=float, default=t.get("lr",            1e-4))
    parser.add_argument("--lr_end_factor", type=float, default=t.get("lr_end_factor", 0.01),
                        help="LR decays linearly to lr * lr_end_factor by the final epoch")
    parser.add_argument("--save_dir", default=t.get("save_dir", "checkpoints_y"))
    parser.add_argument("--resume",   default=None)

    args = parser.parse_args()
    args._model_cfg = cfg.get("model", {})
    return args


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
        scale=args._model_cfg.get("scale", 2),
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
        scale=args._model_cfg.get("scale", 2),
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

    mc = args._model_cfg
    model = EDSREnd2EndModel(
        in_channels=mc.get("in_channels", 1),
        out_channels=mc.get("out_channels", 1),
        num_features=mc.get("num_features", 64),
        num_blocks=mc.get("num_blocks", 8),
        scale=mc.get("scale", 2),
    ).to(device)

    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine_warmup":
        # Linear warmup + CosineAnnealing — SOTA for SR (SwinIR, HAT, BasicVSR++)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup_epochs, 1),
            eta_min=args.min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
        print(f"Scheduler: cosine_warmup (warmup={args.warmup_epochs} epochs, min_lr={args.min_lr:.1e})")
    elif args.scheduler == "multistep":
        milestones = [int(m) for m in args.milestones.split(",")] if args.milestones else [int(args.epochs * 0.5), int(args.epochs * 0.8)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_gamma
        )
        print(f"Scheduler: multistep (milestones={milestones}, gamma={args.lr_gamma})")
    else:  # linear
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.lr_end_factor,
            total_iters=args.epochs,
        )
        print(f"Scheduler: linear (end_factor={args.lr_end_factor})")

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