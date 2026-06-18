"""Evaluate a trained YOLOv8-s SeaShips24790 model and export tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-s on SeaShips24790.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt.")
    parser.add_argument("--data", default="configs/seaships24790.local.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Evaluation split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--project", default="runs/seaships24790_eval", help="Ultralytics eval project.")
    parser.add_argument("--name", default="yolov8s_eval", help="Eval run name.")
    parser.add_argument("--markdown", default="results/yolov8s_eval_summary.md", help="Output Markdown path.")
    parser.add_argument("--csv", default="results/yolov8s_eval_summary.csv", help="Output CSV path.")
    return parser.parse_args()


def safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_metric(metrics, attr_path: str) -> float | None:
    current = metrics
    for attr in attr_path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return None
    return safe_float(current)


def get_class_ap_rows(metrics) -> list[dict[str, object]]:
    names = getattr(metrics, "names", {}) or {}
    box = getattr(metrics, "box", None)
    maps = getattr(box, "maps", None)
    rows: list[dict[str, object]] = []
    if maps is None:
        return rows

    for class_id, ap in enumerate(maps):
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "AP50:95": safe_float(ap),
            }
        )
    return rows


def format_value(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    summary = {
        "Precision": get_metric(metrics, "box.mp"),
        "Recall": get_metric(metrics, "box.mr"),
        "mAP50": get_metric(metrics, "box.map50"),
        "mAP50:95": get_metric(metrics, "box.map"),
    }
    class_rows = get_class_ap_rows(metrics)

    markdown_path = Path(args.markdown)
    csv_path = Path(args.csv)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# YOLOv8-s SeaShips24790 Evaluation Summary",
        "",
        f"- Weights: `{args.weights}`",
        f"- Data: `{args.data}`",
        f"- Split: `{args.split}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {format_value(value)} |")

    lines.extend(["", "## Per-Class AP", "", "| class_id | class_name | AP50:95 |", "|---:|---|---:|"])
    for row in class_rows:
        lines.append(f"| {row['class_id']} | {row['class_name']} | {format_value(row['AP50:95'])} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "class_id", "class_name", "Precision", "Recall", "mAP50", "mAP50:95", "AP50:95"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "class_id": "",
                "class_name": "",
                "Precision": format_value(summary["Precision"]),
                "Recall": format_value(summary["Recall"]),
                "mAP50": format_value(summary["mAP50"]),
                "mAP50:95": format_value(summary["mAP50:95"]),
                "AP50:95": "",
            }
        )
        for row in class_rows:
            writer.writerow(
                {
                    "scope": "class",
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "Precision": "",
                    "Recall": "",
                    "mAP50": "",
                    "mAP50:95": "",
                    "AP50:95": format_value(row["AP50:95"]),
                }
            )

    print("Evaluation summary")
    for key, value in summary.items():
        print(f"{key}: {format_value(value)}")
    print(f"Saved Markdown summary to: {markdown_path}")
    print(f"Saved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()
