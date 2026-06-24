# UNIV SeaShips24790 Detection Experiment

## 1. Experiment objective

Validate the feasibility of using UNIV / ViT representations for downstream object detection on SeaShips24790 and provide a comparison point against existing YOLOv8-s, YOLOv9-s, and YOLOv10-s baselines.

## 2. Current phase

This experiment is a UNIV downstream detection feasibility validation. It is not the final RGB-IR fusion experiment. Semantic-aware PCCL and RGB-IR fusion modules will be implemented separately later.

## 3. Dataset configuration

`configs/seaships24790.yaml` is a safe template only. It intentionally uses `/path/to/SeaShips24790` and must not be edited with server-specific absolute paths before committing.

Create a local dataset config with:

```bash
cp configs/seaships24790.yaml configs/seaships24790.local.yaml
```

Then manually edit `configs/seaships24790.local.yaml` so `path` points to the real SeaShips24790 root. Local YAML files are ignored by Git and should not be committed. If the local YAML is missing, the training/evaluation dataset loader prints:

```text
Data config not found. Please copy configs/seaships24790.yaml to configs/seaships24790.local.yaml and edit the dataset path.
```

SeaShips24790 starts from COCO JSON annotations. Before training, run the existing `convert_coco_to_yolo.py` workflow if YOLO-format labels are not already available. This UNIV detection experiment reads the converted YOLO labels directly and does not modify the verified conversion logic.

Expected split layout under the configured dataset root:

```text
images/train
images/val
images/test
labels/train
labels/val
labels/test
```

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

### Plan A: recommended detection environment

```bash
conda create -n univ_seaships python=3.8 -y
conda activate univ_seaships
pip install -U pip setuptools wheel
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python pillow pyyaml tqdm numpy scipy pandas matplotlib scikit-learn
pip install einops timm==0.4.12 pycocotools torchmetrics
pip install -r requirements.txt
```

### Plan B: legacy UNIV environment if Python 3.8 is incompatible

The original UNIV README says `conda activate UINV`; this appears to be a typo and should be understood as `UNIV`/the created environment name.

```bash
conda create -n univ_legacy python=3.6 -y
conda activate univ_legacy
pip install pip==21.3.1 setuptools==59.6.0 wheel
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Note: `UNIV-main/requirements.txt` pins newer packages such as `torch==2.4.1`, `torchvision==0.19.1`, and `timm==1.0.12`, which conflict with the conservative detection environment above. Prefer installing the explicit Plan A packages first for this detection baseline; use the legacy environment only if the uploaded UNIV code requires it.

The provided evaluator includes a lightweight internal AP implementation, so `pycocotools`/`torchmetrics` are recommended for future metric parity but are not strictly required for the initial smoke test.

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

## 10. Notes

- Do not commit datasets, labels, runs, local YAML files, checkpoints, or `*.pt`/`*.pth` weights.
- UNIV detector results should not be treated as architecturally identical to YOLO results because the detection head is Faster R-CNN, not a YOLO head.
- This experiment mainly validates that UNIV representations can be connected to a detection task.
- Future improvements can include UNIV + YOLO Head, UNIV + FPN, RGB-IR fusion, and UNIV + Semantic-aware PCCL.
