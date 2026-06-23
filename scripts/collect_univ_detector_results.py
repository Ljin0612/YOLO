"""Collect UNIV detector training and evaluation summaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_last_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect UNIV detector result files.")
    parser.add_argument("--train-log", default="results/univ_detector_train_log.csv")
    parser.add_argument("--eval-csv", default="results/univ_detector_eval_summary.csv")
    parser.add_argument("--output-md", default="results/univ_detector_seaships24790_summary.md")
    parser.add_argument("--output-csv", default="results/univ_detector_seaships24790_summary.csv")
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--batch", default="2")
    parser.add_argument("--epochs", default="50")
    parser.add_argument("--freeze-backbone", default="True")
    parser.add_argument("--params", default="")
    parser.add_argument("--notes", default="feasibility baseline")
    args = parser.parse_args()

    train = read_last_row(Path(args.train_log)); eval_row = read_last_row(Path(args.eval_csv))
    row = {
        "Model": "UNIV Detector", "Dataset": "SeaShips24790", "Backbone": "UNIV / ViT", "Detection Head": "Faster R-CNN",
        "imgsz": args.imgsz, "batch": args.batch, "epochs": train.get("epoch", args.epochs), "freeze_backbone": args.freeze_backbone,
        "Precision": eval_row.get("Precision", train.get("Precision", "")), "Recall": eval_row.get("Recall", train.get("Recall", "")),
        "mAP50": eval_row.get("mAP50", train.get("mAP50", "")), "mAP50:95": eval_row.get("mAP50:95", train.get("mAP50:95", "")),
        "Params": args.params, "Notes": args.notes,
    }
    out_md = Path(args.output_md); out_csv = Path(args.output_csv); out_md.parent.mkdir(parents=True, exist_ok=True); out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = list(row.keys())
    out_md.write_text("# UNIV Detector SeaShips24790 Summary\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n| " + " | ".join(str(row[h]) for h in headers) + " |\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader(); writer.writerow(row)
    print(f"Wrote {out_md} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
