# UNIV SeaShips24790 Detection Experiment

## 1. Experiment objective

Validate the feasibility of using UNIV/ViT representations for downstream object detection on SeaShips24790 and provide a comparison point against existing YOLOv8-s, YOLOv9-s, and YOLOv10-s baselines.

## 2. Current phase

This experiment is a UNIV downstream detection feasibility validation. It is not the final RGB-IR fusion experiment. Semantic-aware PCCL and RGB-IR fusion modules will be implemented separately later.

## 3. Dataset format

SeaShips24790 starts from COCO JSON annotations. Before training, run the existing `convert_coco_to_yolo.py` workflow to generate YOLO-format labels. This UNIV detection experiment reads the converted YOLO labels directly and does not modify the verified conversion logic.

## 4. Classes

| id | class |
| ---: | --- |
| 0 | container_ship |
| 1 | passenger_ship |
| 2 | cargo_ship |
| 3 | fishing_boat |
| 4 | island |
| 5 | floatage |

Torchvision Faster R-CNN reserves label `0` for background internally, so the dataset loader shifts object labels to `1..6`.

## 5. Environment installation

```bash
conda create -n univ_seaships python=3.10 -y
conda activate univ_seaships
pip install -r requirements.txt
```

Additional packages typically needed by this baseline:

```bash
pip install torch torchvision pyyaml pillow numpy pycocotools torchmetrics
```

The provided evaluator includes a lightweight internal AP implementation, so `pycocotools`/`torchmetrics` are recommended but not strictly required for the initial smoke test.

## 6. Data check command

```bash
python scripts/check_yolo_dataset.py --data configs/seaships24790.local.yaml
```

## 7. 1 epoch smoke test

```bash
python scripts/train_univ_detector_seaships24790.py \
--data configs/seaships24790.local.yaml \
--imgsz 640 \
--batch 2 \
--epochs 1 \
--device 0 \
--name univ_detector_smoke_test \
--freeze-backbone True \
--amp False
```

## 8. Formal training

```bash
python scripts/train_univ_detector_seaships24790.py \
--data configs/seaships24790.local.yaml \
--univ-weights /path/to/univ_pretrained.pth \
--imgsz 640 \
--batch 2 \
--epochs 50 \
--device 0 \
--name univ_detector_baseline \
--freeze-backbone True \
--amp False
```

## 9. Test evaluation

```bash
python scripts/eval_univ_detector_seaships24790.py \
--weights runs/seaships24790/univ_detector_baseline/best.pth \
--data configs/seaships24790.local.yaml \
--split test \
--imgsz 640 \
--batch 2 \
--device 0
```

## 10. Screen background training

```bash
screen -S univ_seaships
conda activate univ_seaships
python scripts/train_univ_detector_seaships24790.py \
--data configs/seaships24790.local.yaml \
--univ-weights /path/to/univ_pretrained.pth \
--imgsz 640 \
--batch 2 \
--epochs 50 \
--device 0 \
--project runs/seaships24790 \
--name univ_detector_baseline \
--freeze-backbone True \
--amp False
# Detach with Ctrl-a d; reattach with: screen -r univ_seaships
```

## 11. Notes

- Do not commit datasets.
- Do not commit labels.
- Do not commit runs.
- Do not commit `*.pt` or `*.pth` weights.
- UNIV detector results should not be treated as architecturally identical to YOLO results because the detection head is Faster R-CNN, not a YOLO head.
- This experiment mainly validates that UNIV representations can be connected to a detection task.
- Future improvements can include UNIV + YOLO Head, UNIV + FPN, and UNIV + Semantic-aware PCCL.
