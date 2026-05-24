import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.yuv_dataset import YOnlySRDataset
from models.edsr_small import EDSRSmall


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr_yuv", required=True, help="Path to LR FHD YUV420 10-bit file")
    parser.add_argument("--hr_yuv", required=True, help="Path to HR 4K YUV420 10-bit file")

    parser.add_argument("--lr_width", type=int, default=1920)
    parser.add_argument("--lr_height", type=int, default=1080)
    parser.add_argument("--hr_width", type=int, default=3840)
    parser.add_argument("--hr_height", type=int, default=2160)

    parser.add_argument("--num_frames", type=int, required=True)

    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--samples_per_epoch", type=int, default=10000)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_dir", default="checkpoints_y")
    parser.add_argument("--resume", default=None)

    return parser.parse_args()


def train():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = YOnlySRDataset(
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = EDSRSmall(
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

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr_y, hr_y in pbar:
            lr_y = lr_y.to(device)
            hr_y = hr_y.to(device)

            pred_y = model(lr_y)

            loss = criterion(pred_y, hr_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")

        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")


if __name__ == "__main__":
    train()