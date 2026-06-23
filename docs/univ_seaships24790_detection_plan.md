# UNIV SeaShips24790 Detection Plan

## 1. Current UNIV code structure analysis

The original UNIV code is stored under `UNIV-main/`. The relevant model code is the ConvMAE/ViT backbone in `UNIV-main/models/backbone/mcmae/models_convmae.py`; the pretraining entry point is `UNIV-main/pretrain_mcmae.py`. Existing downstream code in `UNIV-main/SEG/MCMAE_SEG/` targets semantic segmentation through MMSegmentation and does not provide a lightweight object detection entry point for SeaShips24790.

## 2. How the UNIV encoder is called

The pretraining script constructs the model with:

```python
models_convmae.__dict__['convmae_convvit_base_patch16']()
```

For detection, the new adapter imports that constructor by adding `UNIV-main/` to `sys.path`, builds the encoder, optionally loads weights, and calls:

```python
latent, attention = encoder(images, mask_ratio=0.0)
```

`mask_ratio=0.0` keeps all image patches for dense downstream detection. Because the uploaded UNIV ConvMAE implementation hard-codes 224-style 14x14 masks and fixed positional embeddings, the adapter internally resizes the Faster R-CNN image batch to 224x224 before calling the encoder while the outer dataset and CLI still use `--imgsz` for loading and detection targets.

## 3. UNIV output tensor shape

`MaskedAutoencoderConvViT.forward()` returns `(latent, attention_map)`. The latent output is patch tokens shaped `[B, N, D]`. For the default ConvMAE base configuration and 224-style input assumptions, `D=768` and `N` is the number of stage-3 patches. The adapter converts square token grids to feature maps as `[B, D, sqrt(N), sqrt(N)]`. If a CLS token is present, it is removed when `N - 1` is square. If the token count is not square, the adapter raises a clear error.

## 4. Implementation plan for this detection experiment

This phase adds a minimal single-scale Faster R-CNN detection baseline:

- Use UNIV/ConvMAE as a frozen-by-default backbone.
- Convert UNIV patch tokens to one feature map.
- Return `OrderedDict[str, Tensor]` with key `"0"` for torchvision detection.
- Use `torchvision.models.detection.FasterRCNN` with `num_classes=7` (background plus six SeaShips24790 classes).
- Keep existing YOLOv8/v9/v10 scripts and COCO-to-YOLO conversion unchanged.
- Read existing YOLO labels directly for PyTorch detection training.

This is a feasibility baseline, not the final RGB-IR fusion or Semantic-aware PCCL experiment.

## 5. New files

- `scripts/datasets/seaships_yolo_detection_dataset.py`
- `scripts/models/univ_detection_adapter.py`
- `scripts/train_univ_detector_seaships24790.py`
- `scripts/eval_univ_detector_seaships24790.py`
- `scripts/collect_univ_detector_results.py`
- `docs/univ_seaships24790_detection_plan.md`
- `docs/univ_seaships24790_detection_experiment.md`
- `docs/seaships24790_detection_comparison_template.md`

## 6. Training and evaluation commands

Smoke test:

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

Formal training:

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

Evaluation:

```bash
python scripts/eval_univ_detector_seaships24790.py \
  --weights runs/seaships24790/univ_detector_baseline/best.pth \
  --data configs/seaships24790.local.yaml \
  --split test \
  --imgsz 640 \
  --batch 2 \
  --device 0
```
