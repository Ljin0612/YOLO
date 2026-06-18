# YOLOv8-s SeaShips24790 Detection Baseline

## Experiment Goal

Run a YOLOv8-s baseline on SeaShips24790 and collect Precision, Recall, mAP50,
mAP50:95, and per-class AP for paper and presentation tables.

## Dataset Layout and Annotation Format

Keep the dataset outside Git. The current SeaShips24790 raw annotations are COCO
JSON files, not native YOLO txt labels. The source dataset is expected to look
like this before conversion:

```text
SeaShips24790/
  images/
    train/
    val/
    test/
  annotations/
    train.json
    val.json
    test.json
```

Convert the COCO JSON annotations to YOLO txt labels locally before checking or
training:

```bash
python scripts/convert_coco_to_yolo.py --dataset-root /path/to/SeaShips24790
```

The converter writes derived label files to:

```text
SeaShips24790/
  labels/
    train/
    val/
    test/
```

Each label file uses YOLO format:

```text
class_id x_center y_center width height
```

All bbox coordinates must be normalized to the 0-1 range. Class IDs must match
this fixed order:

| id | class |
|---:|---|
| 0 | container_ship |
| 1 | passenger_ship |
| 2 | cargo_ship |
| 3 | fishing_boat |
| 4 | island |
| 5 | floatage |

## Local Data Config

`configs/seaships24790.yaml` is a committed template only. Copy it and edit only
the local copy for your machine-specific dataset path:

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

## Recommended Workflow

### 1. Convert COCO JSON to YOLO Labels

```bash
python scripts/convert_coco_to_yolo.py --dataset-root /path/to/SeaShips24790
```

### 2. Check the Converted YOLO Dataset

```bash
python scripts/check_yolo_dataset.py --data configs/seaships24790.local.yaml
```

The report is saved to:

```text
results/dataset_check_report.md
```

### 3. Run a 1 Epoch Smoke Test

```bash
python scripts/train_yolov8s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --epochs 1 \
  --name yolov8s_smoke_test
```

### 4. Run Full Training

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

## Git Hygiene

Do not commit raw datasets, converted labels, training runs, predictions,
weights, checkpoints, experiment logs, or local data YAML files. In particular,
do not commit `SeaShips24790/`, `labels/`, `datasets/`, `data/`, `runs/`,
`weights/`, `checkpoints/`, `wandb/`, `mlruns/`, `*.pt`, `*.pth`, `*.onnx`,
`*.engine`, or `configs/*local*.yaml` / `configs/*local*.yml` files.
