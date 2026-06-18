# YOLOv8-s SeaShips24790 Detection Baseline

## Experiment Goal

Run a YOLOv8-s baseline on SeaShips24790 and collect Precision, Recall, mAP50,
mAP50:95, and per-class AP for paper and presentation tables.

## Dataset Layout

Keep the dataset outside Git. The expected YOLO layout is:

```text
SeaShips24790/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label file must use YOLO format:

```text
class_id x_center y_center width height
```

All bbox coordinates must be normalized to the 0-1 range. Class IDs must match:

| id | class |
|---:|---|
| 0 | container_ship |
| 1 | passenger_ship |
| 2 | cargo_ship |
| 3 | fishing_boat |
| 4 | island |
| 5 | floatage |

## Local Data Config

Copy the committed template and edit only the local copy:

```bash
cp configs/seaships24790.yaml configs/seaships24790.local.yaml
```

Set `path` in `configs/seaships24790.local.yaml` to the real dataset directory.
Local YAML files are ignored by Git and should not be committed.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_yolov8.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements_yolov8.txt
```

## Dataset Check

Run the dataset check before training:

```bash
python scripts/check_yolo_dataset.py --data configs/seaships24790.local.yaml
```

The report is saved to:

```text
results/dataset_check_report.md
```

## 1 Epoch Smoke Test

```bash
python scripts/train_yolov8s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --epochs 1 \
  --name yolov8s_smoke_test
```

## Full Training

```bash
python scripts/train_yolov8s_seaships24790.py \
  --data configs/seaships24790.local.yaml
```

Default training settings:

| setting | value |
|---|---:|
| model | yolov8s.pt |
| imgsz | 640 |
| batch | 8 |
| epochs | 300 |
| optimizer | SGD |
| lr0 | 0.01 |
| seed | 42 |

## Validation or Test Evaluation

```bash
python scripts/eval_yolov8s_seaships24790.py \
  --weights runs/seaships24790/yolov8s_baseline/weights/best.pt \
  --data configs/seaships24790.local.yaml \
  --split test
```

Evaluation summaries are saved to:

```text
results/yolov8s_eval_summary.md
results/yolov8s_eval_summary.csv
```

## Collect Training Results

```bash
python scripts/collect_yolo_results.py \
  --run-dir runs/seaships24790/yolov8s_baseline
```

Collected paper-style tables are saved to:

```text
results/yolov8s_seaships24790_summary.md
results/yolov8s_seaships24790_summary.csv
```

## Notes

Do not commit dataset images, labels, training runs, weights, predictions, or logs.
The repository ignores dataset folders, `runs/`, common weight formats, experiment
tracking folders, and local dataset YAML files.
