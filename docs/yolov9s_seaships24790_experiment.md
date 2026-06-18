# YOLOv9-s SeaShips24790 Experiment

## 实验目的

在 SeaShips24790 上训练 YOLOv9-s baseline，并与 YOLOv8-s、YOLOv10-S、IGC-Net 等结果进行对比。

## 数据集格式说明

SeaShips24790 原始格式是 COCO JSON：

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

训练前需要先运行已有的 COCO to YOLO 转换脚本。转换后使用 `labels/train`、`labels/val`、`labels/test`：

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

## 类别映射说明

YOLO 固定类别顺序为 `container_ship`、`passenger_ship`、`cargo_ship`、`fishing_boat`、`island`、`floatage`。

- COCO id 1 cargo ship -> YOLO id 2 cargo_ship
- COCO id 2 container ship -> YOLO id 0 container_ship
- COCO id 3 fishing boat -> YOLO id 3 fishing_boat
- COCO id 4 passenger ship -> YOLO id 1 passenger_ship
- COCO id 5 island -> YOLO id 4 island
- COCO id 6 flotage -> YOLO id 5 floatage

`flotage` 和 `floatage` 都兼容，并统一映射到 YOLO class id 5。

## 环境安装

```bash
conda activate seaships_yolo
pip install -r requirements_yolov8.txt
```

## 数据转换

```bash
python scripts/convert_coco_to_yolo.py --dataset-root /path/to/SeaShips24790
```

## 数据检查

```bash
python scripts/check_yolo_dataset.py --data configs/seaships24790.local.yaml
```

## 1 epoch smoke test

```bash
python scripts/train_yolov9s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --model yolov9s.pt \
  --epochs 1 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolov9s_smoke_test \
  --amp False
```

## 正式训练

```bash
python scripts/train_yolov9s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --model yolov9s.pt \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolov9s_baseline \
  --amp False
```

如果服务器无法自动下载 `yolov9s.pt`，请先手动下载权重，然后使用本地路径：

```bash
python scripts/train_yolov9s_seaships24790.py \
  --model /home/jinlei/weights/yolov9s.pt \
  --data configs/seaships24790.local.yaml
```

## screen 后台训练

```bash
screen -S yolo9s_seaships
conda activate seaships_yolo
cd /path/to/repo
python scripts/train_yolov9s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --model yolov9s.pt \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolov9s_baseline \
  --amp False
```

启动后按 `Ctrl + A`，then `D` 断开 screen 会话。

## 恢复训练

```bash
python scripts/train_yolov9s_seaships24790.py \
  --model runs/seaships24790/yolov9s_baseline/weights/last.pt \
  --resume True
```

## test 评估

```bash
python scripts/eval_yolov9s_seaships24790.py \
  --weights runs/seaships24790/yolov9s_baseline/weights/best.pt \
  --data configs/seaships24790.local.yaml \
  --split test \
  --device 0
```

## 结果整理

```bash
python scripts/collect_yolov9_results.py \
  --run-dir runs/seaships24790/yolov9s_baseline
```

## 注意事项

- 不要提交数据集。
- 不要提交 labels。
- 不要提交 runs。
- 不要提交 best.pt、last.pt、`*.pt`。
- 如果服务器无法联网下载 `yolov9s.pt`，请手动把 `yolov9s.pt` 放到服务器，并用 `--model` 指定本地路径。
- YOLOv9-s 可能比 YOLOv8-s 训练更慢。
- RTX 2080Ti 单卡建议 `batch=4`。
- 默认 `amp=False`，用于避免 Ultralytics AMP check 在网络不稳定时尝试下载额外权重并超时。
