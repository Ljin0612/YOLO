"""PyTorch detection dataset for SeaShips24790 YOLO labels."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    0: "container_ship",
    1: "passenger_ship",
    2: "cargo_ship",
    3: "fishing_boat",
    4: "island",
    5: "floatage",
}
NUM_CLASSES = 6


def load_data_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "path" not in data:
        raise ValueError(f"Missing required 'path' field in {config_path}")
    root = Path(data["path"]).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    data["path"] = str(root)
    return data


def resolve_split_dir(data: dict[str, Any], split: str) -> Path:
    if split not in data:
        raise ValueError(f"Split '{split}' not found in data YAML. Available keys: {sorted(data)}")
    split_path = Path(str(data[split])).expanduser()
    if not split_path.is_absolute():
        split_path = Path(data["path"]) / split_path
    return split_path


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    x_center, y_center, box_width, box_height = values
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    return [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]


class SeaShipsYoloDetectionDataset(Dataset):
    """Read SeaShips24790 images and YOLO txt labels as torchvision detection targets."""

    def __init__(self, data: str | Path = "configs/seaships24790.local.yaml", split: str = "train", imgsz: int = 640) -> None:
        self.data_path = Path(data).expanduser()
        self.data = load_data_yaml(self.data_path)
        self.split = split
        self.imgsz = int(imgsz)
        self.image_dir = resolve_split_dir(self.data, split)
        self.label_dir = label_dir_from_image_dir(self.image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        self.images = iter_images(self.image_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def label_path_for(self, image_path: Path) -> Path:
        rel_stem = image_path.relative_to(self.image_dir).with_suffix("")
        return self.label_dir / rel_stem.with_suffix(".txt")

    def _read_target(self, label_path: Path, width: int, height: int, scale_x: float, scale_y: float, image_id: int) -> dict[str, torch.Tensor]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            for line_number, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO columns, got {len(parts)}")
                class_id = int(float(parts[0]))
                if class_id < 0 or class_id >= NUM_CLASSES:
                    raise ValueError(f"{label_path}:{line_number}: class_id {class_id} outside 0-{NUM_CLASSES - 1}")
                xyxy = yolo_to_xyxy([float(v) for v in parts[1:]], width, height)
                xyxy = [xyxy[0] * scale_x, xyxy[1] * scale_y, xyxy[2] * scale_x, xyxy[3] * scale_y]
                if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                    boxes.append(xyxy)
                    labels.append(class_id + 1)  # torchvision reserves 0 for background.
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)
        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }

    def __getitem__(self, index: int):
        image_path = self.images[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            image = F.to_tensor(img)
        target = self._read_target(self.label_path_for(image_path), orig_w, orig_h, self.imgsz / orig_w, self.imgsz / orig_h, index)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test SeaShips24790 detection dataset loading.")
    parser.add_argument("--data", default="configs/seaships24790.local.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    dataset = SeaShipsYoloDetectionDataset(args.data, args.split, args.imgsz)
    image, target = dataset[0]
    print(f"images={len(dataset)} image_shape={tuple(image.shape)} boxes={target['boxes'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
