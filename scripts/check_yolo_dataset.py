"""Check a YOLO-format SeaShips24790 dataset before training.

The script validates split directories, image/label pairing, label row format,
class IDs, normalized bounding boxes, and basic object-size statistics. It also
writes a Markdown report that can be kept with experiment notes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
DEFAULT_CLASS_COUNT = 6
DEFAULT_INPUT_SIZE = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SeaShips24790 YOLO dataset folders and labels."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a YOLO data YAML, e.g. configs/seaships24790.local.yaml.",
    )
    parser.add_argument(
        "--output",
        default="results/dataset_check_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Input size used for rough small/medium/large area statistics.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=DEFAULT_CLASS_COUNT,
        help="Expected number of classes.",
    )
    return parser.parse_args()


def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "path" not in data:
        raise ValueError(f"Missing required 'path' field in {path}")
    return data


def resolve_split_dir(dataset_root: Path, split_value: str, expected_leaf: str) -> Path:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    if split_path.name == expected_leaf:
        return split_path
    return split_path


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def relative_stem(path: Path, root: Path) -> Path:
    return path.relative_to(root).with_suffix("")


def classify_box_size(width: float, height: float, imgsz: int) -> str:
    area = width * imgsz * height * imgsz
    if area < 32 * 32:
        return "small"
    if area <= 96 * 96:
        return "medium"
    return "large"


def validate_label_file(
    label_path: Path,
    num_classes: int,
    imgsz: int,
) -> tuple[int, Counter[int], Counter[str], list[str], bool]:
    box_count = 0
    class_counts: Counter[int] = Counter()
    size_counts: Counter[str] = Counter()
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return box_count, class_counts, size_counts, errors, True

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_number}: expected 5 columns, got {len(parts)}")
            continue

        try:
            class_id_float = float(parts[0])
            class_id = int(class_id_float)
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric label value")
            continue

        if class_id_float != class_id:
            errors.append(f"{label_path}:{line_number}: class_id must be an integer")
            continue

        if not 0 <= class_id < num_classes:
            errors.append(f"{label_path}:{line_number}: class_id {class_id} outside 0-{num_classes - 1}")

        if any(value < 0 or value > 1 for value in values):
            errors.append(f"{label_path}:{line_number}: bbox values must be normalized to 0-1")

        _, _, width, height = values
        if width <= 0 or height <= 0:
            errors.append(f"{label_path}:{line_number}: bbox width and height must be greater than 0")

        box_count += 1
        class_counts[class_id] += 1
        size_counts[classify_box_size(width, height, imgsz)] += 1

    return box_count, class_counts, size_counts, errors, False


def format_counter(counter: Counter, keys: Iterable) -> str:
    return ", ".join(f"{key}: {counter.get(key, 0)}" for key in keys)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    data = load_data_yaml(data_path)
    dataset_root = Path(data["path"]).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()

    class_names = data.get("names", {})
    if isinstance(class_names, list):
        class_names = {index: name for index, name in enumerate(class_names)}

    report_lines = [
        "# SeaShips24790 Dataset Check Report",
        "",
        f"- Data config: `{data_path}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Input size for area bins: `{args.imgsz}`",
        "",
    ]
    all_errors: list[str] = []
    total_boxes = 0
    total_class_counts: Counter[int] = Counter()
    total_size_counts: Counter[str] = Counter()

    for split in SPLITS:
        image_dir = resolve_split_dir(dataset_root, data.get(split, ""), "images")
        label_dir = label_dir_from_image_dir(image_dir)
        split_errors: list[str] = []
        split_class_counts: Counter[int] = Counter()
        split_size_counts: Counter[str] = Counter()
        split_boxes = 0
        empty_label_files = 0

        if not image_dir.exists():
            split_errors.append(f"Missing image directory: {image_dir}")
            images: list[Path] = []
        else:
            images = iter_images(image_dir)

        if not label_dir.exists():
            split_errors.append(f"Missing label directory: {label_dir}")
            labels: list[Path] = []
        else:
            labels = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())

        image_stems = {relative_stem(path, image_dir): path for path in images} if image_dir.exists() else {}
        label_stems = {relative_stem(path, label_dir): path for path in labels} if label_dir.exists() else {}

        missing_labels = sorted(set(image_stems) - set(label_stems))
        extra_labels = sorted(set(label_stems) - set(image_stems))
        if missing_labels:
            split_errors.append(f"Images without labels: {len(missing_labels)}")
        if extra_labels:
            split_errors.append(f"Labels without images: {len(extra_labels)}")

        for stem in sorted(set(image_stems) & set(label_stems)):
            box_count, class_counts, size_counts, errors, is_empty = validate_label_file(
                label_stems[stem], args.num_classes, args.imgsz
            )
            split_boxes += box_count
            split_class_counts.update(class_counts)
            split_size_counts.update(size_counts)
            split_errors.extend(errors)
            empty_label_files += int(is_empty)

        total_boxes += split_boxes
        total_class_counts.update(split_class_counts)
        total_size_counts.update(split_size_counts)
        all_errors.extend(f"[{split}] {error}" for error in split_errors)

        report_lines.extend(
            [
                f"## {split}",
                "",
                f"- Images directory: `{image_dir}`",
                f"- Labels directory: `{label_dir}`",
                f"- Image files: {len(images)}",
                f"- Label files: {len(labels)}",
                f"- Object boxes: {split_boxes}",
                f"- Empty label files: {empty_label_files}",
                f"- Missing label files: {len(missing_labels)}",
                f"- Extra label files: {len(extra_labels)}",
                f"- Object sizes: {format_counter(split_size_counts, ('small', 'medium', 'large'))}",
                "",
                "| class_id | class_name | objects |",
                "|---:|---|---:|",
            ]
        )
        for class_id in range(args.num_classes):
            report_lines.append(
                f"| {class_id} | {class_names.get(class_id, '')} | {split_class_counts.get(class_id, 0)} |"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "## Overall",
            "",
            f"- Total object boxes: {total_boxes}",
            f"- Object sizes: {format_counter(total_size_counts, ('small', 'medium', 'large'))}",
            "",
            "| class_id | class_name | objects |",
            "|---:|---|---:|",
        ]
    )
    for class_id in range(args.num_classes):
        report_lines.append(
            f"| {class_id} | {class_names.get(class_id, '')} | {total_class_counts.get(class_id, 0)} |"
        )

    if all_errors:
        report_lines.extend(["", "## Errors", ""])
        report_lines.extend(f"- {error}" for error in all_errors[:200])
        if len(all_errors) > 200:
            report_lines.append(f"- ... truncated {len(all_errors) - 200} additional errors")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nSaved Markdown report to: {output_path}")

    if all_errors:
        print(f"\nDataset check failed with {len(all_errors)} issue(s).")
        return 1

    print("\nDataset check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
