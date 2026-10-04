"""Train a ResNet34 U-Net on Kvasir-SEG.

Example:
    python -m polypseg.train --data-dir data --out-dir runs/unet_resnet34
    python -m polypseg.train --config configs/default.yaml --override train.epochs=10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from polypseg.config import load_config
from polypseg.data import KvasirSegDataset, list_pairs, make_splits
from polypseg.metrics import BCEDiceLoss, segmentation_metrics
from polypseg.model import ResNetUNet, count_parameters
from polypseg.viz import per_sample_dice, save_curves, save_prediction_grid

METRIC_NAMES = ("dice", "iou", "precision", "recall")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    if warmup_epochs == 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, desc) -> float:
    model.train()
    total, count = 0.0, 0
    for images, masks, _ in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss = criterion(logits.float(), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, threshold: float = 0.5) -> dict[str, float]:
    """Mean of per-image metrics over a loader (every image weighs the same)."""
    model.eval()
    scores = {name: [] for name in METRIC_NAMES}
    loss_sum, count = 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            loss_sum += criterion(logits, masks).item() * images.size(0)
        for name, values in segmentation_metrics(logits, masks, threshold).items():
            scores[name].append(values.cpu())
        count += images.size(0)
    result = {name: torch.cat(values).mean().item() for name, values in scores.items()}
    if criterion is not None:
        result["loss"] = loss_sum / count
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", default="data", help="Folder containing Kvasir-SEG/")
    parser.add_argument("--out-dir", default="runs/unet_resnet34")
    parser.add_argument("--config", default=None, help="Optional YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. train.epochs=10"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device_name} | AMP: {use_amp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------------------------
    pairs = list_pairs(args.data_dir)
    splits = make_splits(pairs, cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["seed"])
    (out_dir / "splits.json").write_text(
        json.dumps({k: [img.name for img, _ in v] for k, v in splits.items()}, indent=2)
    )
    print({k: len(v) for k, v in splits.items()})

    img_size = cfg["data"]["img_size"]
    datasets = {
        name: KvasirSegDataset(items, img_size=img_size, train=(name == "train"))
        for name, items in splits.items()
    }
    batch_size = cfg["train"]["batch_size"]
    workers = cfg["train"]["num_workers"]
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        drop_last=len(datasets["train"]) > batch_size,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        **loader_kwargs,
    )
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    # ---- model / optimisation ---------------------------------------------------------------
    model = ResNetUNet(pretrained=cfg["model"]["pretrained"]).to(device)
    criterion = BCEDiceLoss(cfg["train"]["bce_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    epochs = cfg["train"]["epochs"]
    scheduler = build_scheduler(optimizer, epochs, cfg["train"]["warmup_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- training loop ----------------------------------------------------------------------
    history: list[dict] = []
    best_dice, best_epoch, best_val = -1.0, 0, {}
    best_path = out_dir / "best.pt"
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            desc=f"epoch {epoch}/{epochs}",
        )
        val = evaluate(model, val_loader, device, criterion)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_dice": val["dice"],
                "val_iou": val["iou"],
                "lr": lr,
            }
        )
        print(
            f"epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} | val loss {val['loss']:.4f}"
            f" | val Dice {val['dice']:.4f} | val IoU {val['iou']:.4f} | lr {lr:.2e}"
        )

        if val["dice"] > best_dice:
            best_dice, best_epoch, best_val = val["dice"], epoch, val
            torch.save(
                {
                    "model": model.state_dict(),
                    "img_size": img_size,
                    "epoch": epoch,
                    "val_dice": val["dice"],
                    "config": cfg,
                },
                best_path,
            )

    train_minutes = (time.time() - start) / 60

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, out_dir / "curves.png")

    # ---- final evaluation on the held-out test split, using the best-validation checkpoint ----
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, criterion)

    dices = per_sample_dice(model, datasets["test"], device)
    rng = random.Random(cfg["seed"])
    random_idx = rng.sample(range(len(dices)), k=min(6, len(dices)))
    worst_idx = sorted(range(len(dices)), key=dices.__getitem__)[:6]
    save_prediction_grid(
        model,
        datasets["test"],
        device,
        random_idx,
        out_dir / "samples.png",
        title="Random test images (fixed seed, not cherry-picked)",
    )
    save_prediction_grid(
        model,
        datasets["test"],
        device,
        worst_idx,
        out_dir / "failure_cases.png",
        title="Lowest-Dice test images",
    )

    results = {
        "best_epoch": best_epoch,
        "validation": best_val,
        "test": test,
        "test_dice_median": float(np.median(dices)),
        "test_dice_std": float(np.std(dices)),
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "parameters": count_parameters(model),
        "train_minutes": round(train_minutes, 1),
        "device": device_name,
        "torch": torch.__version__,
        "config": cfg,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"\nTest metrics (best-validation checkpoint, epoch {best_epoch}):")
    for name in METRIC_NAMES:
        print(f"  {name:9s} {test[name]:.4f}")
    print(f"Artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
