# YOLOv10-s SeaShips24790 Experiment

## 1. 实验目的

在 SeaShips24790 上训练 YOLOv10-s detection baseline，并与 YOLOv8-s、YOLOv9-s 结果进行对比。YOLOv10-s 应与已有 baseline 使用相同的数据划分、输入尺寸、batch size 和训练 epoch 设置，保证对比公平。

## 2. 数据集格式说明

SeaShips24790 原始格式使用 COCO JSON 标注：

```text
SeaShips24790/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

训练 YOLO 前需要先运行已有 COCO-to-YOLO 转换脚本。转换后使用以下标签目录：

```text
SeaShips24790/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 3. 类别映射说明

YOLO 固定类别顺序为：

```text
0 container_ship
1 passenger_ship
2 cargo_ship
3 fishing_boat
4 island
5 floatage
```

COCO 到 YOLO 的类别映射如下：

| COCO category | YOLO class |
| --- | --- |
| COCO id 1 cargo ship | YOLO id 2 cargo_ship |
| COCO id 2 container ship | YOLO id 0 container_ship |
| COCO id 3 fishing boat | YOLO id 3 fishing_boat |
| COCO id 4 passenger ship | YOLO id 1 passenger_ship |
| COCO id 5 island | YOLO id 4 island |
| COCO id 6 flotage | YOLO id 5 floatage |

`flotage` 和 `floatage` 都应兼容，并统一映射到 YOLO class id 5 (`floatage`)。

## 4. 环境安装

建议创建独立环境：

```bash
conda activate yolov10_seaships
```

也可以复用已有 `seaships_yolo` 或 `yolov9_seaships` 环境，前提是其中的 `ultralytics` 版本支持 YOLOv10。

## 5. 数据转换命令

```bash
python scripts/convert_coco_to_yolo.py --dataset-root /path/to/SeaShips24790
```

## 6. 数据检查命令

```bash
python scripts/check_yolo_dataset.py --data configs/seaships24790.local.yaml
```

## 7. 1 epoch smoke test 命令

```bash
python scripts/train_yolov10s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --model yolov10s.pt \
  --epochs 1 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolov10s_smoke_test \
  --amp False
```

如果服务器无法联网自动下载权重，请先手动下载 `yolov10s.pt`，再使用 `--model /home/jinlei/weights/yolov10s.pt` 指定本地路径。

## 8. 正式训练命令

```bash
python scripts/train_yolov10s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --model yolov10s.pt \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolov10s_baseline \
  --amp False
```

默认 `project=runs/seaships24790`、`name=yolov10s_baseline`，训练脚本会在训练结束后从 `model.trainer.save_dir` 打印真实保存目录，以及 `best.pt`、`last.pt`、`results.csv` 的路径。

## 9. screen 后台训练命令

```bash
screen -S yolo10s_seaships
conda activate yolov10_seaships
cd /path/to/repo
python scripts/train_yolov10s_seaships24790.py \
  --data configs/seaships24790.local.yaml \
  --model yolov10s.pt \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolov10s_baseline \
  --amp False
```

Detach screen session:

```text
Ctrl + A, then D
```

## 10. 恢复训练命令

```bash
python scripts/train_yolov10s_seaships24790.py \
  --model runs/seaships24790/yolov10s_baseline/weights/last.pt \
  --resume True
```

## 11. test 评估命令

```bash
python scripts/eval_yolov10s_seaships24790.py \
  --weights runs/seaships24790/yolov10s_baseline/weights/best.pt \
  --data configs/seaships24790.local.yaml \
  --split test \
  --device 0
```

评估结果默认保存到：

```text
results/yolov10s_eval_summary.md
results/yolov10s_eval_summary.csv
```

## 12. 结果整理命令

```bash
python scripts/collect_yolov10_results.py \
  --run-dir runs/seaships24790/yolov10s_baseline
```

整理结果默认保存到：

```text
results/yolov10s_seaships24790_summary.md
results/yolov10s_seaships24790_summary.csv
```

## 13. 注意事项

- 不要提交数据集。
- 不要提交 labels。
- 不要提交 runs。
- 不要提交 `best.pt`、`last.pt`、`*.pt` 或其他权重文件。
- 如果服务器无法联网下载 `yolov10s.pt`，请手动把 `yolov10s.pt` 放到服务器，并用 `--model` 指定本地路径。
- YOLOv10-s 训练结果应与 YOLOv8-s、YOLOv9-s 使用相同数据划分、`imgsz`、`batch`、`epochs`，保证对比公平。
- RTX 2080Ti 单卡建议 `batch=4`。
- 默认 `amp=False`，以适配服务器网络不稳定时 Ultralytics AMP check 可能下载超时的问题。
