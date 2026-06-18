"""Collect YOLO training metrics into paper-ready Markdown and CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_ALIASES = {
    "Precision": ["metrics/precision(B)", "precision", "Precision"],
    "Recall": ["metrics/recall(B)", "recall", "Recall"],
    "mAP50": ["metrics/mAP50(B)", "mAP50", "map50"],
    "mAP50:95": ["metrics/mAP50-95(B)", "mAP50:95", "map"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect YOLO results.csv into summary tables.")
    parser.add_argument("--run-dir", default="runs/seaships24790/yolov8s_baseline", help="YOLO run directory.")
    parser.add_argument("--model", default="YOLOv8-s", help="Model name for the table.")
    parser.add_argument("--dataset", default="SeaShips24790", help="Dataset name for the table.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Planned training epochs.")
    parser.add_argument("--prefer", choices=("best", "last"), default="best", help="Select best mAP50:95 row or last row.")
    parser.add_argument("--markdown", default="results/yolov8s_seaships24790_summary.md", help="Output Markdown path.")
    parser.add_argument("--csv", default="results/yolov8s_seaships24790_summary.csv", help="Output CSV path.")
    return parser.parse_args()


def find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {column.strip(): column for column in frame.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def select_row(frame: pd.DataFrame, prefer: str) -> pd.Series:
    map_col = find_column(frame, METRIC_ALIASES["mAP50:95"])
    if prefer == "best" and map_col is not None:
        return frame.loc[frame[map_col].astype(float).idxmax()]
    return frame.iloc[-1]


def metric_value(row: pd.Series, frame: pd.DataFrame, name: str) -> float | str:
    column = find_column(frame, METRIC_ALIASES[name])
    if column is None:
        return ""
    return row[column]


def markdown_table(row: dict[str, object]) -> str:
    headers = list(row.keys())
    values = [str(row[key]) for key in headers]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            "| " + " | ".join(values) + " |",
        ]
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.csv"
    weights_path = run_dir / "weights" / "best.pt"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing YOLO results CSV: {results_path}")

    frame = pd.read_csv(results_path)
    frame.columns = [column.strip() for column in frame.columns]
    row = select_row(frame, args.prefer)
    epochs_done = int(row["epoch"]) if "epoch" in frame.columns else len(frame)

    summary = {
        "Model": args.model,
        "Dataset": args.dataset,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": epochs_done,
        "Precision": metric_value(row, frame, "Precision"),
        "Recall": metric_value(row, frame, "Recall"),
        "mAP50": metric_value(row, frame, "mAP50"),
        "mAP50:95": metric_value(row, frame, "mAP50:95"),
        "weights_path": str(weights_path),
        "results_path": str(results_path),
    }

    markdown_path = Path(args.markdown)
    csv_path = Path(args.csv)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    table = markdown_table(summary)
    markdown_path.write_text("# YOLOv8-s SeaShips24790 Result Summary\n\n" + table + "\n", encoding="utf-8")

    print(table)
    print(f"\nSaved Markdown summary to: {markdown_path}")
    print(f"Saved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()
