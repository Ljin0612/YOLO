"""UNIV encoder with a lightweight YOLOv8-style anchor-free detection head.

The module intentionally lives next to, but does not modify, the existing UNIV
Faster R-CNN adapter.  It reuses the UNIV ConvMAE encoder and adds a small
multi-scale, decoupled, anchor-free detection head inspired by YOLOv8.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
UNIV_ROOT = REPO_ROOT / "UNIV-main"
if str(UNIV_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIV_ROOT))

import models.backbone.mcmae.models_convmae as models_convmae


def build_univ_encoder() -> nn.Module:
    """Build the ConvMAE/UNIV encoder used by the SeaShips experiments."""
    return models_convmae.__dict__["convmae_convvit_base_patch16"]()


def load_univ_weights(encoder: nn.Module, weights: str | None) -> None:
    """Load a UNIV pretraining checkpoint such as pretrained/checkpoint0400.pth."""
    if not weights:
        LOGGER.warning("No --univ-weights provided; the UNIV encoder is randomly initialized.")
        return
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type at {weights}: {type(checkpoint)!r}")

    state: dict[str, Any] = checkpoint
    for key in ("student", "model", "state_dict", "teacher", "backbone"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break

    cleaned: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        name = raw_name
        for prefix in ("module.backbone.", "backbone.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        cleaned[name] = value
    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    LOGGER.info("Loaded UNIV weights from %s (missing=%d unexpected=%d)", weights, len(missing), len(unexpected))


def tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    """Convert ViT tokens into a BCHW feature map, dropping CLS when present."""
    token_count = tokens.shape[1]
    if int(math.isqrt(token_count)) ** 2 != token_count:
        tokens = tokens[:, 1:, :]
    batch, token_count, channels = tokens.shape
    side = int(math.isqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"UNIV token count {token_count} cannot be reshaped to a square map")
    return tokens.transpose(1, 2).contiguous().reshape(batch, channels, side, side)


class ConvBnSilu(nn.Sequential):
    """Conv-BN-SiLU block matching the style of YOLO family heads."""

    def __init__(self, c1: int, c2: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(c1, c2, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class YOLOv8StyleHead(nn.Module):
    """Decoupled box/class branches for one feature-map scale."""

    def __init__(self, channels: int, nc: int) -> None:
        super().__init__()
        self.box = nn.Sequential(ConvBnSilu(channels, channels), ConvBnSilu(channels, channels), nn.Conv2d(channels, 4, 1))
        self.cls = nn.Sequential(ConvBnSilu(channels, channels), ConvBnSilu(channels, channels), nn.Conv2d(channels, nc, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output layout is [ltrb box distances, class logits], with no anchor/objectness channel.
        return torch.cat((self.box(x), self.cls(x)), dim=1)


class UNIVYOLOv8Detector(nn.Module):
    """UNIV backbone plus YOLOv8-style multi-scale anchor-free detection head."""

    strides = (8, 16, 32)

    def __init__(
        self,
        nc: int = 6,
        univ_weights: str | None = "pretrained/checkpoint0400.pth",
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.encoder = build_univ_encoder()
        load_univ_weights(self.encoder, univ_weights)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        if unfreeze_last_blocks > 0:
            for block in list(getattr(self.encoder, "blocks3", []))[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        self.adapters = nn.ModuleList(ConvBnSilu(768, head_channels) for _ in self.strides)
        self.heads = nn.ModuleList(YOLOv8StyleHead(head_channels, nc) for _ in self.strides)

    def backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out = self.encoder(x, mask_ratio=0.0)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        return tokens_to_feature_map(out) if out.ndim == 3 else out

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = self.backbone_features(x)
        image_h, image_w = x.shape[-2:]
        sizes = [(max(1, image_h // stride), max(1, image_w // stride)) for stride in self.strides]
        return [
            head(adapter(F.interpolate(feat, size=size, mode="bilinear", align_corners=False)))
            for adapter, head, size in zip(self.adapters, self.heads, sizes)
        ]


def decode_predictions(
    preds: list[torch.Tensor],
    imgsz: int | tuple[int, int],
    nc: int = 6,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[dict[str, torch.Tensor]]:
    """Decode YOLOv8-style l/t/r/b distance predictions into torchvision detections."""
    image_h, image_w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    outputs: list[dict[str, torch.Tensor]] = []
    for batch_index in range(preds[0].shape[0]):
        boxes_all: list[torch.Tensor] = []
        scores_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        for pred, stride in zip(preds, UNIVYOLOv8Detector.strides):
            p = pred[batch_index]
            height, width = p.shape[-2:]
            distances = F.softplus(p[:4])
            cls_logits = p[4 : 4 + nc]
            yy, xx = torch.meshgrid(torch.arange(height, device=p.device), torch.arange(width, device=p.device), indexing="ij")
            cx = (xx.float() + 0.5) * stride
            cy = (yy.float() + 0.5) * stride
            boxes = torch.stack(
                (
                    (cx - distances[0] * stride).clamp(0, image_w).reshape(-1),
                    (cy - distances[1] * stride).clamp(0, image_h).reshape(-1),
                    (cx + distances[2] * stride).clamp(0, image_w).reshape(-1),
                    (cy + distances[3] * stride).clamp(0, image_h).reshape(-1),
                ),
                dim=1,
            )
            scores, labels = cls_logits.sigmoid().reshape(nc, -1).max(dim=0)
            keep = scores > conf_thres
            if keep.any():
                boxes_all.append(boxes[keep])
                scores_all.append(scores[keep])
                labels_all.append(labels[keep] + 1)
        if boxes_all:
            boxes = torch.cat(boxes_all)
            scores = torch.cat(scores_all)
            labels = torch.cat(labels_all)
            keep_indices = batched_nms(boxes, scores, labels, iou_thres)[:max_det]
            outputs.append({"boxes": boxes[keep_indices], "scores": scores[keep_indices], "labels": labels[keep_indices]})
        else:
            outputs.append(
                {
                    "boxes": torch.zeros((0, 4), device=preds[0].device),
                    "scores": torch.zeros(0, device=preds[0].device),
                    "labels": torch.zeros(0, dtype=torch.long, device=preds[0].device),
                }
            )
    return outputs


def build_univ_yolov8_detector(**kwargs: Any) -> UNIVYOLOv8Detector:
    return UNIVYOLOv8Detector(**kwargs)
