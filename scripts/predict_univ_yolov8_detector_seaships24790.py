"""Run image prediction with UNIV + YOLOv8-style SeaShips detector."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.seaships_yolo_detection_dataset import CLASS_NAMES
from scripts.models.univ_yolov8_detector import build_univ_yolov8_detector, decode_predictions

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with UNIV + YOLOv8-style detector on SeaShips24790 images.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--save-dir", default="runs/detect/univ_yolov8_seaships24790/predict")
    return parser.parse_args()


def select_device(device: str) -> torch.device:
    return torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else f"cuda:{device}")


def iter_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_univ_yolov8_detector(nc=6, univ_weights=None).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval()

    for image_path in iter_sources(Path(args.source)):
        with Image.open(image_path) as image:
            original = image.convert("RGB")
        resized = original.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        tensor = F.to_tensor(resized).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = decode_predictions(model(tensor), args.imgsz, nc=model.nc, conf_thres=args.conf_thres, iou_thres=args.iou_thres)[0]
        draw = ImageDraw.Draw(resized)
        for box, score, label in zip(pred["boxes"].cpu(), pred["scores"].cpu(), pred["labels"].cpu()):
            x1, y1, x2, y2 = [float(v) for v in box]
            name = CLASS_NAMES.get(int(label) - 1, str(int(label)))
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), f"{name} {float(score):.2f}", fill="red")
        out_path = save_dir / image_path.name
        resized.save(out_path)
        print(f"saved {out_path} detections={len(pred['boxes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
