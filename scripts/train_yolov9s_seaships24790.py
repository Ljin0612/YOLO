"""Train a YOLOv9-s baseline on SeaShips24790 with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def str2bool(value: bool | str) -> bool:
    """Parse common command-line boolean spellings."""
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv9-s on SeaShips24790.")
    parser.add_argument("--data", default="configs/seaships24790.local.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", default="yolov9s.pt", help="YOLOv9-s checkpoint path or model name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size; default is conservative for RTX 2080Ti.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or comma-separated ids.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project", default="runs/seaships24790", help="Ultralytics project directory.")
    parser.add_argument("--name", default="yolov9s_baseline", help="Run name.")
    parser.add_argument("--amp", type=str2bool, default=False, help="Enable AMP; defaults to False to avoid online AMP checks.")
    parser.add_argument("--resume", type=str2bool, default=False, help="Resume training from the provided checkpoint.")
    return parser.parse_args()


def warn_if_missing_local_model(model_path: str) -> None:
    """Warn when a user supplied local-looking checkpoint does not exist."""
    path = Path(model_path).expanduser()
    if path.suffix == ".pt" and not path.exists():
        print(
            "Note: model checkpoint was not found in the current filesystem: "
            f"{model_path}\n"
            "Ultralytics may try to download it automatically. If this server cannot "
            "access the internet, manually download yolov9s.pt and pass its local path "
            "with --model."
        )


def main() -> None:
    args = parse_args()
    warn_if_missing_local_model(args.model)

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
        amp=args.amp,
        resume=args.resume,
    )

    trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    save_dir = Path(trainer_save_dir or getattr(results, "save_dir", Path(args.project) / args.name))
    print("\nTraining finished.")
    print(f"Save directory: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    print(f"Training metrics CSV: {save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
