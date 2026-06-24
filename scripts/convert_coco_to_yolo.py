"""Convert SeaShips24790 COCO JSON annotations to YOLO label files.

The converter keeps the six SeaShips classes in a fixed YOLO class order and
creates one label file per image, including empty files for images with no
valid annotations.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


YOLO_CLASS_NAMES = (
    "container_ship",
    "passenger_ship",
    "cargo_ship",
    "fishing_boat",
    "island",
    "floatage",
)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO txt labels.")
    parser.add_argument("--dataset-root", required=True, help="SeaShips24790 dataset root directory.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), help="Dataset splits to convert.")
    parser.add_argument("--annotations-dir", default="annotations", help="COCO annotation directory under dataset root.")
    parser.add_argument("--images-dir", default="images", help="Image directory under dataset root.")
    parser.add_argument("--labels-dir", default="labels", help="Output label directory under dataset root.")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    """Normalize COCO category names for robust name-based matching."""
    return " ".join(name.strip().lower().replace("_", " ").split())


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"Missing required COCO field '{key}' in {path}")
    return data


def image_label_path(image: dict[str, Any], label_dir: Path, images_dir: str, split: str) -> Path:
    file_name = Path(str(image.get("file_name", "")))
    relative_stem = file_name.with_suffix("")
    leading_parts = list(relative_stem.parts)
    if leading_parts[:2] == [images_dir, split]:
        relative_stem = Path(*leading_parts[2:])
    elif leading_parts[:1] == [split]:
        relative_stem = Path(*leading_parts[1:])
    return label_dir / relative_stem.with_suffix(".txt")


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


def convert_bbox(bbox: list[float], image_width: float, image_height: float) -> tuple[float, float, float, float, bool]:
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    clipped_x_min = min(max(x_min, 0.0), image_width)
    clipped_y_min = min(max(y_min, 0.0), image_height)
    clipped_x_max = min(max(x_max, 0.0), image_width)
    clipped_y_max = min(max(y_max, 0.0), image_height)
    clipped = (clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max) != (x_min, y_min, x_max, y_max)

    norm_x_center = ((clipped_x_min + clipped_x_max) / 2.0) / image_width
    norm_y_center = ((clipped_y_min + clipped_y_max) / 2.0) / image_height
    norm_width = (clipped_x_max - clipped_x_min) / image_width
    norm_height = (clipped_y_max - clipped_y_min) / image_height

    values = tuple(clip(value) for value in (norm_x_center, norm_y_center, norm_width, norm_height))
    return values[0], values[1], values[2], values[3], clipped


def category_mapping(categories: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, str], list[str]]:
    name_to_yolo = {normalize_name(name): index for index, name in enumerate(YOLO_CLASS_NAMES)}
    name_to_yolo.update(
        {
            "container ship": 0,
            "passenger ship": 1,
            "cargo ship": 2,
            "fishing boat": 3,
            "flotage": 5,
            "floatage": 5,
        }
    )

    coco_id_to_name: dict[int, str] = {}
    coco_to_yolo: dict[int, int] = {}
    warnings_list: list[str] = []
    for category in categories:
        category_id = category.get("id")
        original_name = str(category.get("name", ""))
        normalized_name = normalize_name(original_name)
        if category_id is None:
            warnings_list.append(f"category without id skipped: {category}")
            continue

        int_category_id = int(category_id)
        coco_id_to_name[int_category_id] = original_name
        yolo_class_id = name_to_yolo.get(normalized_name)
        if yolo_class_id is None:
            warnings_list.append(f"category id {category_id} name '{category.get('name')}' is outside fixed classes")
            continue
        coco_to_yolo[int_category_id] = yolo_class_id
    return coco_to_yolo, coco_id_to_name, warnings_list


def print_category_mapping(split: str, coco_to_yolo: dict[int, int], coco_id_to_name: dict[int, str]) -> None:
    print(f"\nSplit: {split}")
    print("  Category mapping (COCO id -> COCO name -> YOLO class id):")
    for category_id in sorted(coco_id_to_name):
        yolo_class_id = coco_to_yolo.get(category_id)
        yolo_value = "unsupported" if yolo_class_id is None else str(yolo_class_id)
        print(f"    {category_id} -> {coco_id_to_name[category_id]} -> {yolo_value}")


def convert_split(dataset_root: Path, split: str, annotations_dir: str, images_dir: str, labels_dir: str) -> None:
    annotation_path = dataset_root / annotations_dir / f"{split}.json"
    label_dir = dataset_root / labels_dir / split
    data = load_coco(annotation_path)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = {image.get("id"): image for image in data["images"]}
    label_lines: dict[Any, list[str]] = defaultdict(list)
    coco_to_yolo, coco_id_to_name, category_warnings = category_mapping(data["categories"])
    print_category_mapping(split, coco_to_yolo, coco_id_to_name)
    class_counts: Counter[int] = Counter()
    skipped = 0
    clipped_boxes = 0

    for message in category_warnings:
        warnings.warn(f"[{split}] {message}")

    for annotation in data["annotations"]:
        image_id = annotation.get("image_id")
        image = images.get(image_id)
        if image is None:
            skipped += 1
            warnings.warn(f"[{split}] skipped annotation {annotation.get('id')}: image_id {image_id} not found")
            continue

        category_id = annotation.get("category_id")
        try:
            int_category_id = int(category_id)
        except (TypeError, ValueError):
            int_category_id = -1
        class_id = coco_to_yolo.get(int_category_id)
        if class_id is None:
            skipped += 1
            category_name = coco_id_to_name.get(int_category_id, "unknown")
            warnings.warn(
                f"[{split}] skipped annotation {annotation.get('id')}: "
                f"unsupported category_id {category_id} name '{category_name}'"
            )
            continue

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped += 1
            warnings.warn(f"[{split}] skipped annotation {annotation.get('id')}: invalid bbox {bbox}")
            continue

        x_min, y_min, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            skipped += 1
            warnings.warn(f"[{split}] skipped annotation {annotation.get('id')}: non-positive bbox size {bbox}")
            continue

        image_width = float(image.get("width", 0))
        image_height = float(image.get("height", 0))
        if image_width <= 0 or image_height <= 0:
            skipped += 1
            warnings.warn(f"[{split}] skipped annotation {annotation.get('id')}: invalid image size")
            continue

        norm_x, norm_y, norm_w, norm_h, clipped = convert_bbox(
            [x_min, y_min, width, height], image_width, image_height
        )
        if norm_w <= 0 or norm_h <= 0:
            skipped += 1
            warnings.warn(f"[{split}] skipped annotation {annotation.get('id')}: bbox outside image after clipping")
            continue
        if clipped:
            clipped_boxes += 1
            warnings.warn(f"[{split}] clipped annotation {annotation.get('id')}: bbox exceeded image bounds")

        label_lines[image_id].append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        class_counts[class_id] += 1

    label_file_count = 0
    for image in data["images"]:
        label_path = image_label_path(image, label_dir, images_dir, split)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = label_lines.get(image.get("id"), [])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        label_file_count += 1

    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Label files: {label_file_count}")
    print("  Class objects:")
    for class_id, class_name in enumerate(YOLO_CLASS_NAMES):
        print(f"    {class_id} {class_name}: {class_counts.get(class_id, 0)}")
    print(f"  Skipped abnormal annotations: {skipped}")
    print(f"  Clipped bounding boxes: {clipped_boxes}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    for split in args.splits:
        convert_split(dataset_root, split, args.annotations_dir, args.images_dir, args.labels_dir)


if __name__ == "__main__":
    main()
