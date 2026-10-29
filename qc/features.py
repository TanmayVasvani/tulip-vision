"""Frozen ImageNet backbone -> patch features (the only torch code path).

Features are layer2 + upsampled layer3 of a ResNet, concatenated and
smoothed over each 3x3 neighbourhood (PatchCore's local aggregation). With
resnet18 that is a 384-dim vector for every 8x8-pixel cell of the model
input: fine enough to see a single missing tuft, cheap enough for a CPU.

Weights
    "imagenet"   torchvision's ImageNet weights, downloaded once on the
                 training machine and cached by torch
    <file.pth>   a local ResNet state dict (torchvision or timm naming), for
                 a training machine without internet
    None         random init (tests only)

A trained bundle stores the backbone weights it was built with, so the
inspection PC never needs internet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models

from .data import atomic_replace

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    # ~4x heavier. For subtle defects try a larger input first (--input-long-side
    # 576); compare both on your own held-out report before switching backbone.
    "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights"),
}
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
_USED_PREFIXES = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.")
_EXPORT_FORMAT = "tulip-vision/extractor@1"


class WeightsError(RuntimeError):
    pass


def _read_state_dict(path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(blob.get(key), dict):
                blob = blob[key]
                break
    out = {}
    for k, v in blob.items():
        if not torch.is_tensor(v):
            continue
        for prefix in ("module.", "backbone.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = v
    return out


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        input_size=(256, 256),
        weights="imagenet",
        device: str | None = None,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; options: {sorted(BACKBONES)}")
        ctor, weights_enum = BACKBONES[backbone]

        if weights == "imagenet":
            try:
                net = ctor(weights=getattr(models, weights_enum).IMAGENET1K_V1)
            except (OSError, RuntimeError) as exc:
                raise WeightsError(
                    f"could not download the ImageNet {backbone} weights ({exc}). Connect this machine to "
                    "the internet once (torch caches them), or pass --weights <a local .pth file>."
                ) from exc
            self.weights_desc = f"torchvision {weights_enum}.IMAGENET1K_V1"
        elif weights in (None, "none", "random"):
            net = ctor(weights=None)
            self.weights_desc = "random init (tests only)"
        else:
            net = ctor(weights=None)
            sd = _read_state_dict(weights)
            own = net.state_dict()
            needed = [
                k for k in own if k.startswith(_USED_PREFIXES) and not k.endswith("num_batches_tracked")
            ]
            missing = [k for k in needed if k not in sd]
            if missing:
                raise ValueError(
                    f"{weights}: {len(missing)} backbone tensors missing (e.g. {missing[:2]}) — "
                    f"is this a {backbone} checkpoint?"
                )
            bad = [k for k in needed if tuple(sd[k].shape) != tuple(own[k].shape)]
            if bad:
                raise ValueError(f"{weights}: shape mismatch for {bad[:2]} — wrong architecture?")
            net.load_state_dict({k: sd[k] for k in needed}, strict=False)
            self.weights_desc = f"file {Path(weights).name}"

        self.backbone = backbone
        self.input_size = (int(input_size[0]), int(input_size[1]))  # (width, height)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()
        self.to(self.device)

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _out_channels(layer) -> int:
        block = layer[-1]  # BasicBlock ends in conv2, Bottleneck in conv3
        last = block.conv3 if hasattr(block, "conv3") else block.conv2
        return int(last.out_channels)

    @property
    def feature_dim(self) -> int:
        return self._out_channels(self.layer2) + self._out_channels(self.layer3)

    stride = 8  # one feature cell per 8x8 pixels of model input (layer2)

    # ------------------------------------------------------------- forward
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """ROI-cropped PIL image -> normalised [3, H, W] tensor (CPU)."""
        img = img.convert("RGB").resize(self.input_size, Image.Resampling.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (t - _MEAN) / _STD

    @torch.inference_mode()
    def features(self, batch: torch.Tensor):
        """[B, 3, H, W] -> (patch features [B, P, D] float32 numpy, grid (h, w))."""
        x = self.layer1(self.stem(batch.to(self.device)))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        b, c, h, w = f.shape
        feats = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return feats.float().cpu().numpy(), (h, w)

    def extract(self, images, batch_size: int = 8):
        """Yield (features [P, D], grid) for each ROI-cropped PIL image, in order."""
        images = list(images)
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(im) for im in images[i : i + batch_size]])
            feats, grid = self.features(batch)
            for f in feats:
                yield f, grid

    # ---------------------------------------------------------- save / load
    def export(self) -> dict:
        return {
            "format": _EXPORT_FORMAT,
            "backbone": self.backbone,
            "input_size": list(self.input_size),
            "weights": self.weights_desc,
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.export(), tmp)
        atomic_replace(tmp, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "FeatureExtractor":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != _EXPORT_FORMAT:
            raise ValueError(f"{path} is not a Tulip Vision extractor file")
        fx = cls(blob["backbone"], tuple(blob["input_size"]), weights=None, device=device)
        fx.load_state_dict(blob["state_dict"])
        fx.weights_desc = blob.get("weights", "unknown")
        fx.eval()
        return fx
