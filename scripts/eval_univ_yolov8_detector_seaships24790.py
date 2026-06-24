"""Evaluate UNIV + YOLOv8-style detector on SeaShips24790."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.seaships_yolo_detection_dataset import CLASS_NAMES, SeaShipsYoloDetectionDataset, collate_fn
from scripts.eval_univ_detector_seaships24790 import evaluate_model
from scripts.models.univ_yolov8_detector import build_univ_yolov8_detector, decode_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UNIV + YOLOv8-style detector on SeaShips24790.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/seaships24790.local.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect/univ_yolov8_seaships24790")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--markdown", default="results/univ_yolov8_detector_eval_summary.md")
    parser.add_argument("--csv", default="results/univ_yolov8_detector_eval_summary.csv")
    return parser.parse_args()


def select_device(device: str) -> torch.device:
    return torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else f"cuda:{device}")


class EvalWrapper(torch.nn.Module):
    def __init__(self, model, imgsz: int, conf_thres: float, iou_thres: float) -> None:
        super().__init__()
        self.model = model
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def eval(self):
        self.model.eval()
        return self

    def forward(self, images):
        x = torch.stack(images) if isinstance(images, (list, tuple)) else images
        return decode_predictions(self.model(x), self.imgsz, nc=self.model.nc, conf_thres=self.conf_thres, iou_thres=self.iou_thres)


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = SeaShipsYoloDetectionDataset(args.data, args.split, args.imgsz)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_univ_yolov8_detector(nc=6, univ_weights=None).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    metrics = evaluate_model(EvalWrapper(model, args.imgsz, args.conf_thres, args.iou_thres), loader, device, 6)

    markdown_path = Path(args.markdown)
    csv_path = Path(args.csv)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# UNIV YOLOv8-style Detector Evaluation Summary",
        "",
        "- Model: UNIV YOLOv8-style Detector",
        "- Dataset: SeaShips24790",
        "- Backbone: UNIV / ViT",
        "- Detection Head: YOLOv8-style anchor-free decoupled head",
        f"- Weights path: `{args.weights}`",
        f"- Split: `{args.split}`",
        f"- imgsz: `{args.imgsz}`",
        f"- Classes: `{', '.join(CLASS_NAMES.values())}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.6f} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(metrics)
    print(f"save_dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
