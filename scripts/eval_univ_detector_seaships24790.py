"""Evaluate a UNIV Faster R-CNN detector on SeaShips24790."""
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
from scripts.models.univ_detection_adapter import build_univ_faster_rcnn


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def compute_ap(scores: list[float], matches: list[int], num_gt: int) -> float:
    if num_gt == 0 or not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = torch.tensor([matches[i] for i in order], dtype=torch.float32)
    fp = 1 - tp
    tp_cum = torch.cumsum(tp, 0); fp_cum = torch.cumsum(fp, 0)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-7)
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    return float(torch.trapz(mpre, mrec))


def evaluate_model(model, data_loader, device: torch.device, num_classes: int = 6) -> dict[str, float]:
    model.eval()
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    per_threshold = {thr: {c: {"scores": [], "matches": [], "gt": 0} for c in range(1, num_classes + 1)} for thr in thresholds}
    tp50 = fp50 = gt_total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].cpu(); pred_labels = output["labels"].cpu(); pred_scores = output["scores"].cpu()
                gt_boxes = target["boxes"].cpu(); gt_labels = target["labels"].cpu(); gt_total += int(gt_boxes.shape[0])
                for cls in range(1, num_classes + 1):
                    pidx = torch.where(pred_labels == cls)[0]; gidx = torch.where(gt_labels == cls)[0]
                    pboxes = pred_boxes[pidx]; pscores = pred_scores[pidx]; gboxes = gt_boxes[gidx]
                    for thr in thresholds:
                        bucket = per_threshold[thr][cls]; bucket["gt"] += int(gboxes.shape[0])
                        used: set[int] = set(); order = torch.argsort(pscores, descending=True)
                        ious = box_iou(pboxes, gboxes)
                        for oi in order.tolist():
                            match = 0
                            if gboxes.numel() > 0:
                                best_iou, best_j = torch.max(ious[oi], dim=0)
                                if float(best_iou) >= thr and int(best_j) not in used:
                                    match = 1; used.add(int(best_j))
                            bucket["scores"].append(float(pscores[oi])); bucket["matches"].append(match)
                            if thr == 0.5:
                                tp50 += match; fp50 += 1 - match
    ap50_by_class = {CLASS_NAMES[c - 1]: compute_ap(per_threshold[0.5][c]["scores"], per_threshold[0.5][c]["matches"], per_threshold[0.5][c]["gt"]) for c in range(1, num_classes + 1)}
    map50 = sum(ap50_by_class.values()) / num_classes
    aps = []
    for thr in thresholds:
        aps.extend(compute_ap(per_threshold[thr][c]["scores"], per_threshold[thr][c]["matches"], per_threshold[thr][c]["gt"]) for c in range(1, num_classes + 1))
    return {"Precision": tp50 / max(tp50 + fp50, 1), "Recall": tp50 / max(gt_total, 1), "mAP50": map50, "mAP50:95": sum(aps) / max(len(aps), 1), **{f"AP50/{k}": v for k, v in ap50_by_class.items()}}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNIV detector on SeaShips24790.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/seaships24790.local.yaml")
    parser.add_argument("--univ-weights", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="univ_detector_eval")
    parser.add_argument("--markdown", default="results/univ_detector_eval_summary.md")
    parser.add_argument("--csv", default="results/univ_detector_eval_summary.csv")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def main() -> int:
    args = parse_args(); device = select_device(args.device)
    dataset = SeaShipsYoloDetectionDataset(args.data, args.split, args.imgsz)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    model = build_univ_faster_rcnn(args.univ_weights).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))
    metrics = evaluate_model(model, loader, device)
    md_path = Path(args.markdown); csv_path = Path(args.csv); md_path.parent.mkdir(parents=True, exist_ok=True); csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# UNIV Detector Evaluation Summary", "", "- Model: UNIV Detector", "- Dataset: SeaShips24790", "- Backbone: UNIV / ViT", "- Detection Head: Faster R-CNN", f"- Weights path: `{args.weights}`", f"- UNIV weights path: `{args.univ_weights}`", f"- Split: `{args.split}`", f"- imgsz: `{args.imgsz}`", f"- batch: `{args.batch}`", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys())); writer.writeheader(); writer.writerow(metrics)
    print(f"Markdown: {md_path}\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
