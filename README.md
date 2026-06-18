# SeaShips24790 YOLO Experiments

This repository is used for maritime object detection experiments on the SeaShips24790 dataset.

Initial goal:
- Run YOLOv8-s baseline on SeaShips24790.
- Record mAP50, mAP50:95, precision, recall and per-class AP.
- Prepare for future comparison with YOLOv9-S, YOLOv10-S, YOLO11 and UNIV-based models.

Dataset:
- The dataset files are not stored in this repository.
- Users should configure the local dataset path in `configs/seaships24790.yaml`.

Classes:
0. container_ship
1. passenger_ship
2. cargo_ship
3. fishing_boat
4. island
5. floatage# YOLO
Experiments for maritime object detection on SeaShips24790, including YOLOv8-s baseline and future UNIV-based downstream evaluations.
