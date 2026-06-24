"""Train UNIV + YOLOv8-style detector on SeaShips24790."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.seaships_yolo_detection_dataset import SeaShipsYoloDetectionDataset, collate_fn
from scripts.eval_univ_detector_seaships24790 import evaluate_model as evaluate_torchvision
from scripts.models.univ_yolov8_detector import UNIVYOLOv8Detector, build_univ_yolov8_detector, decode_predictions


def str2bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNIV + YOLOv8-style detector on SeaShips24790.")
    parser.add_argument("--data", default="configs/seaships24790.local.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640, choices=[320, 640])
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect/univ_yolov8_seaships24790")
    parser.add_argument("--name", default="train")
    parser.add_argument("--univ-weights", default="pretrained/checkpoint0400.pth")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--freeze-backbone", type=str2bool, default=True)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def select_device(device: str) -> torch.device:
    return torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else f"cuda:{device}")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def yolov8_loss(preds: list[torch.Tensor], targets, nc: int = 6) -> dict[str, torch.Tensor]:
    """Small smoke-test friendly YOLOv8-style assignment/loss.

    Each GT box is assigned to the cell containing its center at every scale.
    This is intentionally simple so the experiment can smoke-test the UNIV
    backbone and new detection head without changing the Faster R-CNN path.
    """
    device = preds[0].device
    loss_box = torch.tensor(0.0, device=device)
    loss_cls = torch.tensor(0.0, device=device)

    for pred, stride in zip(preds, UNIVYOLOv8Detector.strides):
        batch, _, height, width = pred.shape
        cls_target = torch.zeros((batch, nc, height, width), device=device)
        box_target = torch.zeros((batch, 4, height, width), device=device)
        pos = torch.zeros((batch, 1, height, width), dtype=torch.bool, device=device)
        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = (target["labels"].to(device) - 1).clamp(0, nc - 1)
            for box, label in zip(boxes, labels):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                gx = torch.clamp((cx / stride).long(), 0, width - 1)
                gy = torch.clamp((cy / stride).long(), 0, height - 1)
                cls_target[batch_index, label, gy, gx] = 1.0
                pos[batch_index, 0, gy, gx] = True
                anchor_x = (gx.float() + 0.5) * stride
                anchor_y = (gy.float() + 0.5) * stride
                box_target[batch_index, :, gy, gx] = torch.stack(
                    ((anchor_x - box[0]) / stride, (anchor_y - box[1]) / stride, (box[2] - anchor_x) / stride, (box[3] - anchor_y) / stride)
                ).clamp(min=0)
        loss_cls = loss_cls + F.binary_cross_entropy_with_logits(pred[:, 4 : 4 + nc], cls_target)
        if pos.any():
            box_mask = pos.expand_as(pred[:, :4])
            loss_box = loss_box + F.smooth_l1_loss(F.softplus(pred[:, :4])[box_mask], box_target[box_mask])
    return {"loss_box": loss_box, "loss_cls": loss_cls, "loss_total": loss_box * 5.0 + loss_cls}


class EvalWrapper(torch.nn.Module):
    def __init__(self, model: UNIVYOLOv8Detector, imgsz: int) -> None:
        super().__init__()
        self.model = model
        self.imgsz = imgsz

    def eval(self):
        self.model.eval()
        return self

    def forward(self, images):
        x = torch.stack(images) if isinstance(images, (list, tuple)) else images
        return decode_predictions(self.model(x), self.imgsz, nc=self.model.nc, conf_thres=0.001, iou_thres=0.6)


def append_csv(path: Path, row: dict[str, float]) -> None:
    old_rows: list[dict[str, str]] = []
    fields: list[str] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            old_rows = list(reader)
    fields += [key for key in row if key not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    set_seed()
    device = select_device(args.device)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.csv"

    train_ds = SeaShipsYoloDetectionDataset(args.data, "train", args.imgsz)
    val_ds = SeaShipsYoloDetectionDataset(args.data, "val", args.imgsz)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_univ_yolov8_detector(
        nc=6,
        univ_weights=args.univ_weights,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    best_map50 = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_map50 = float(checkpoint.get("best_map50", -1.0))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        steps = 0
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            x = torch.stack([image.to(device) for image in images])
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                losses = yolov8_loss(model(x), targets, nc=model.nc)
                loss = losses["loss_total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu())
            if batch_index % 10 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} batch {batch_index}/{len(train_loader)} loss_total={totals['loss_total'] / steps:.4f}", flush=True)

        averages = {key: value / max(steps, 1) for key, value in totals.items()}
        metrics = evaluate_torchvision(EvalWrapper(model, args.imgsz), val_loader, device, 6)
        row = {"epoch": epoch + 1, **averages, **{key: metrics.get(key, 0.0) for key in ["Precision", "Recall", "mAP50", "mAP50:95"]}}
        print(
            f"Epoch {epoch + 1}: loss_total={row['loss_total']:.4f} loss_box={row['loss_box']:.4f} "
            f"loss_cls={row['loss_cls']:.4f} mAP50={row['mAP50']:.4f} mAP50:95={row['mAP50:95']:.4f}",
            flush=True,
        )
        append_csv(log_path, row)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "best_map50": best_map50, "args": vars(args)}
        torch.save(checkpoint, save_dir / "last.pth")
        if row["mAP50"] > best_map50:
            best_map50 = row["mAP50"]
            checkpoint["best_map50"] = best_map50
            torch.save(checkpoint, save_dir / "best.pth")
    print(f"save_dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
