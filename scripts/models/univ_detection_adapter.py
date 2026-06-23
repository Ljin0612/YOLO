"""UNIV encoder adapter for torchvision Faster R-CNN detection."""
from __future__ import annotations

import logging
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

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
        LOGGER.warning("No --univ-weights provided; UNIV encoder is randomly initialized and is only suitable for smoke tests.")
        return
    checkpoint = torch.load(weights, map_location="cpu")
    state: dict[str, Any]
    if isinstance(checkpoint, dict):
        for key in ("student", "model", "state_dict", "teacher", "backbone"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        else:
            state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type at {weights}: {type(checkpoint)!r}")
    cleaned = {}
    for name, value in state.items():
        for prefix in ("module.backbone.", "backbone.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned[name] = value
    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    LOGGER.info("Loaded UNIV weights from %s (missing=%d unexpected=%d)", weights, len(missing), len(unexpected))


def _maybe_drop_cls_token(tokens: torch.Tensor) -> torch.Tensor:
    n = tokens.shape[1]
    if int(math.isqrt(n)) ** 2 == n:
        return tokens
    if n > 1 and int(math.isqrt(n - 1)) ** 2 == n - 1:
        return tokens[:, 1:, :]
    raise ValueError(f"Cannot reshape UNIV token output with N={n} to square feature map; pass square input or customize adapter.")


def tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    tokens = _maybe_drop_cls_token(tokens)
    b, n, d = tokens.shape
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"Patch token count {n} is not square after CLS handling.")
    return tokens.transpose(1, 2).contiguous().reshape(b, d, side, side)


class UNIVDetectionBackbone(nn.Module):
    """Wrap UNIV/ConvMAE encoder as a single-scale Faster R-CNN backbone."""

    def __init__(self, univ_weights: str | None = None, freeze_backbone: bool = True, unfreeze_last_blocks: int = 0) -> None:
        super().__init__()
        self.encoder = build_univ_encoder()
        load_univ_weights(self.encoder, univ_weights)
        self.out_channels = 768
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        if unfreeze_last_blocks > 0:
            blocks = getattr(self.encoder, "blocks3", [])
            for block in list(blocks)[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for name in ("norm", "patch_embed4"):
                module = getattr(self.encoder, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        # UNIV ConvMAE was authored around 224x224 inputs and fixed 14x14 token masks.
        # Faster R-CNN can still consume the resulting feature map while preserving
        # target boxes in the outer detection transform.
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # UNIV ConvMAE forward returns (tokens [B,N,D], attention_map). mask_ratio=0 keeps all patches.
        output = self.encoder(x, mask_ratio=0.0)
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        if output.ndim == 3:
            output = tokens_to_feature_map(output)
        elif output.ndim != 4:
            raise ValueError(f"Unsupported UNIV encoder output shape: {tuple(output.shape)}")
        return OrderedDict([("0", output)])


def build_univ_faster_rcnn(univ_weights: str | None = None, num_classes: int = 7, freeze_backbone: bool = True, unfreeze_last_blocks: int = 0) -> FasterRCNN:
    backbone = UNIVDetectionBackbone(univ_weights=univ_weights, freeze_backbone=freeze_backbone, unfreeze_last_blocks=unfreeze_last_blocks)
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, min_size=224, max_size=224)
    return model
