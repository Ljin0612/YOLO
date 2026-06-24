"""UNIV backbone with a lightweight YOLO-style anchor-free detection head."""
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

try:
    import models.backbone.mcmae.models_convmae as models_convmae
except ImportError as exc:  # pragma: no cover
    models_convmae = None
    _UNIV_IMPORT_ERROR = exc
else:
    _UNIV_IMPORT_ERROR = None


def build_univ_encoder() -> nn.Module:
    if models_convmae is None:
        raise ImportError(f"Could not import UNIV ConvMAE modules from {UNIV_ROOT}: {_UNIV_IMPORT_ERROR}")
    return models_convmae.__dict__["convmae_convvit_base_patch16"]()


def load_univ_weights(encoder: nn.Module, weights: str | None) -> None:
    if not weights:
        LOGGER.warning("No --univ-weights provided; UNIV encoder is randomly initialized.")
        return
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type at {weights}: {type(checkpoint)!r}")
    state: dict[str, Any] = checkpoint
    for key in ("student", "model", "state_dict", "teacher", "backbone"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break
    cleaned = {}
    for name, value in state.items():
        for prefix in ("module.backbone.", "backbone.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned[name] = value
    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    LOGGER.info("Loaded UNIV weights from %s (missing=%d unexpected=%d)", weights, len(missing), len(unexpected))


def tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    n = tokens.shape[1]
    if int(math.isqrt(n)) ** 2 != n:
        tokens = tokens[:, 1:, :]
    b, n, d = tokens.shape
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"UNIV token count {n} cannot be reshaped to a square map")
    return tokens.transpose(1, 2).contiguous().reshape(b, d, side, side)


class ConvBnSilu(nn.Sequential):
    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__(nn.Conv2d(c1, c2, k, padding=k // 2, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))


class DecoupledHead(nn.Module):
    def __init__(self, channels: int, nc: int) -> None:
        super().__init__()
        self.cls = nn.Sequential(ConvBnSilu(channels, channels), ConvBnSilu(channels, channels), nn.Conv2d(channels, nc, 1))
        self.reg = nn.Sequential(ConvBnSilu(channels, channels), ConvBnSilu(channels, channels), nn.Conv2d(channels, 4, 1))
        self.obj = nn.Sequential(ConvBnSilu(channels, channels), nn.Conv2d(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.reg(x), self.obj(x), self.cls(x)), dim=1)


class UNIVYoloDetector(nn.Module):
    """YOLO-style detector returning raw feature predictions or post-NMS detections."""

    strides = (8, 16, 32)

    def __init__(self, nc: int = 6, univ_weights: str | None = None, freeze_backbone: bool = True, unfreeze_last_blocks: int = 0, head_channels: int = 128) -> None:
        super().__init__()
        self.nc = nc
        self.encoder = build_univ_encoder()
        load_univ_weights(self.encoder, univ_weights)
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if unfreeze_last_blocks > 0:
            for block in list(getattr(self.encoder, "blocks3", []))[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        self.adapters = nn.ModuleList([ConvBnSilu(768, head_channels), ConvBnSilu(768, head_channels), ConvBnSilu(768, head_channels)])
        self.heads = nn.ModuleList([DecoupledHead(head_channels, nc) for _ in self.strides])

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
        h, w = x.shape[-2:]
        sizes = [(max(1, h // s), max(1, w // s)) for s in self.strides]
        return [head(adapter(F.interpolate(feat, size=size, mode="bilinear", align_corners=False))) for adapter, head, size in zip(self.adapters, self.heads, sizes)]


def decode_predictions(preds: list[torch.Tensor], imgsz: int | tuple[int, int], nc: int = 6, conf_thres: float = 0.25, iou_thres: float = 0.45) -> list[dict[str, torch.Tensor]]:
    ih, iw = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    outputs: list[dict[str, torch.Tensor]] = []
    for b in range(preds[0].shape[0]):
        boxes_all=[]; scores_all=[]; labels_all=[]
        for pred, stride in zip(preds, UNIVYoloDetector.strides):
            p = pred[b]
            h, w = p.shape[-2:]
            reg = F.softplus(p[:4]); obj = p[4:5].sigmoid(); cls = p[5:5+nc].sigmoid()
            yy, xx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing="ij")
            cx = (xx.float() + 0.5) * stride; cy = (yy.float() + 0.5) * stride
            x1 = (cx - reg[0] * stride).clamp(0, iw); y1 = (cy - reg[1] * stride).clamp(0, ih)
            x2 = (cx + reg[2] * stride).clamp(0, iw); y2 = (cy + reg[3] * stride).clamp(0, ih)
            scores, labels = (obj * cls).reshape(nc, -1).max(0)
            keep = scores > conf_thres
            if keep.any():
                boxes_all.append(torch.stack((x1.reshape(-1), y1.reshape(-1), x2.reshape(-1), y2.reshape(-1)), 1)[keep])
                scores_all.append(scores[keep]); labels_all.append(labels[keep] + 1)
        if boxes_all:
            boxes = torch.cat(boxes_all); scores = torch.cat(scores_all); labels = torch.cat(labels_all)
            keep = batched_nms(boxes, scores, labels, iou_thres)[:300]
            outputs.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        else:
            outputs.append({"boxes": torch.zeros((0,4), device=preds[0].device), "scores": torch.zeros(0, device=preds[0].device), "labels": torch.zeros(0, dtype=torch.long, device=preds[0].device)})
    return outputs


def build_univ_yolo_detector(**kwargs: Any) -> UNIVYoloDetector:
    return UNIVYoloDetector(**kwargs)
