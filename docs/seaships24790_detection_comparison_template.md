# SeaShips24790 Detection Comparison Template

| Model         | Backbone         | Detection Head | Dataset       | imgsz | batch | epochs | Precision | Recall | mAP50 | mAP50:95 | Params | FLOPs | Notes                |
| ------------- | ---------------- | -------------- | ------------- | ----: | ----: | -----: | --------: | -----: | ----: | -------: | -----: | ----: | -------------------- |
| YOLOv8-s      | CSPDarknet-like  | YOLO Head      | SeaShips24790 |   640 |     4 |    300 |           |        |       |          |        |       | baseline             |
| YOLOv9-s      | YOLOv9 Backbone  | YOLO Head      | SeaShips24790 |   640 |     4 |    300 |           |        |       |          |        |       | baseline             |
| YOLOv10-s     | YOLOv10 Backbone | YOLO Head      | SeaShips24790 |   640 |     4 |    300 |           |        |       |          |        |       | baseline             |
| UNIV Detector | UNIV / ViT       | Faster R-CNN   | SeaShips24790 |   640 |     2 |     50 |           |        |       |          |        |       | feasibility baseline |
