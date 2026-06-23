"""Train a UNIV Faster R-CNN detector on SeaShips24790."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.seaships_yolo_detection_dataset import SeaShipsYoloDetectionDataset, collate_fn
from scripts.models.univ_detection_adapter import build_univ_faster_rcnn
from scripts.eval_univ_detector_seaships24790 import evaluate_model


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNIV detector on SeaShips24790.")
    parser.add_argument("--data", default="configs/seaships24790.local.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--univ-weights", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--project", default="runs/seaships24790")
    parser.add_argument("--name", default="univ_detector_baseline")
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--freeze-backbone", type=str2bool, default=True)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args(); set_seed(args.seed)
    device = select_device(args.device)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / "univ_detector_train_log.csv"

    train_ds = SeaShipsYoloDetectionDataset(args.data, args.split, args.imgsz)
    val_ds = SeaShipsYoloDetectionDataset(args.data, args.val_split, args.imgsz)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda")

    model = build_univ_faster_rcnn(args.univ_weights, freeze_backbone=args.freeze_backbone, unfreeze_last_blocks=args.unfreeze_last_blocks).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch = 0; best_map50 = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)); best_map50 = float(ckpt.get("best_map50", -1.0))

    fields = ["epoch", "loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "Precision", "Recall", "mAP50", "mAP50:95"]
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        model.train(); totals = {k: 0.0 for k in fields[1:5]}; steps = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            steps += 1; totals["loss"] += float(loss.detach().cpu())
            for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
                totals[key] += float(loss_dict.get(key, torch.tensor(0.0)).detach().cpu())
        train_metrics = {k: v / max(steps, 1) for k, v in totals.items()}
        eval_metrics = evaluate_model(model, val_loader, device, num_classes=6)
        row = {"epoch": epoch + 1, **train_metrics, **{k: eval_metrics.get(k, 0.0) for k in fields[-4:]}}
        print("Epoch {epoch}: loss_classifier={loss_classifier:.4f} loss_box_reg={loss_box_reg:.4f} loss_objectness={loss_objectness:.4f} loss_rpn_box_reg={loss_rpn_box_reg:.4f} mAP50={mAP50:.4f}".format(**row))
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "best_map50": best_map50, "args": vars(args)}
        torch.save(ckpt, save_dir / "last.pth")
        if row["mAP50"] > best_map50:
            best_map50 = row["mAP50"]; ckpt["best_map50"] = best_map50; torch.save(ckpt, save_dir / "best.pth")

    print(f"save_dir: {save_dir}\nbest.pth: {save_dir / 'best.pth'}\nlast.pth: {save_dir / 'last.pth'}\ntrain log path: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
