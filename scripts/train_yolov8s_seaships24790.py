"""Train a YOLOv8-s baseline on SeaShips24790 with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8-s on SeaShips24790.")
    parser.add_argument("--data", default="configs/seaships24790.local.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model checkpoint or YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or comma-separated ids.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project", default="runs/seaships24790", help="Ultralytics project directory.")
    parser.add_argument("--name", default="yolov8s_baseline", help="Run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr0,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    print("\nTraining finished.")
    print(f"Best weights should be saved at: {save_dir / 'weights' / 'best.pt'}")
    print(f"Training metrics CSV should be saved at: {save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
