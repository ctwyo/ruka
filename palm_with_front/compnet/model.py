"""
CompNet model architecture
Source: https://github.com/xuliangcs/compnet
Paper:  "CompNet: Competitive Neural Network for Palmprint Recognition
         Using Learnable Gabor Kernels" (IEEE SPL 2021)

Inference helpers added for our pipeline.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ── Learnable Gabor Convolution (LGC) ─────────────────────────────────────────
class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super().__init__()
        assert channel_in == 1
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio if init_ratio > 0 else 1.0

        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]))
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]))
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False,
        )
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen_kernel(self):
        ksize = self.kernel_size
        xmax = ksize // 2
        x0 = torch.arange(-xmax, xmax + 1, device=self.sigma.device).float()
        y0 = x0.clone()
        y = y0.view(1, -1).repeat(self.channel_out, self.channel_in, ksize, 1)
        x = x0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, ksize)
        theta = self.theta.view(-1, 1, 1, 1)
        x_t = x * torch.cos(theta) + y * torch.sin(theta)
        y_t = -x * torch.sin(theta) + y * torch.cos(theta)
        sigma = self.sigma.view(-1, 1, 1, 1)
        f = self.f.view(-1, 1, 1, 1)
        psi = self.psi.view(-1, 1, 1, 1)
        gb = -torch.exp(-0.5 * ((self.gamma * x_t) ** 2 + y_t ** 2) / (8 * sigma ** 2)) \
             * torch.cos(2 * math.pi * f * x_t + psi)
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self._gen_kernel()
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        assert channel_in == 1
        self.gabor_conv2d = GaborConv2d(1, n_competitor, ksize, stride, padding, init_ratio)
        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.argmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x = (x - self.b) * self.a
        x = self.argmax(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x


class ArcMarginProduct(nn.Module):
    """ArcFace head. We only need it to load weights — never used at inference."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


class compnet(nn.Module):
    """
    Output of getFeatureCode(x): (B, 512) L2-normalized feature vector.
    Input: (B, 1, 128, 128) grayscale tensor (per-image normalized).
    """
    def __init__(self, num_classes=600):
        super().__init__()
        self.num_classes = num_classes
        self.cb1 = CompetitiveBlock(1, 9, ksize=35, stride=3, padding=0, init_ratio=1)
        self.cb2 = CompetitiveBlock(1, 9, ksize=17, stride=3, padding=0, init_ratio=0.5)
        self.cb3 = CompetitiveBlock(1, 9, ksize=7,  stride=3, padding=0, init_ratio=0.25)
        self.fc = nn.Linear(9708, 512)
        self.drop = nn.Dropout(p=0.25)
        self.arclayer = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=False)

    def getFeatureCode(self, x):
        x1 = self.cb1(x).flatten(1)
        x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


# ── inference-only preprocessing (matches NormSingleROI from upstream) ────────
_CLAHE = None   # lazy-init, cv2 not imported at module load


def preprocess_palm(roi_bgr: np.ndarray, mask: np.ndarray | None = None,
                    size: int = 128, debug_dir: str | None = None) -> torch.Tensor:
    """
    BGR uint8 ROI from OpenCV  →  (1, 1, 128, 128) float tensor.
    Replicates `T.Resize(128) → T.ToTensor() → NormSingleROI(1)`.

    `mask` (uint8, same H×W as roi_bgr): 255 inside palm, 0 outside.
    Background pixels are zeroed AFTER CLAHE (so contrast is computed on
    the full crop) and AFTER resize (so the boundary stays a hard zero).
    NormSingleROI then ignores zero pixels when computing mean/std, so the
    background no longer leaks into the embedding.

    If debug_dir is provided, each pipeline stage is dumped as a PNG there.
    """
    import cv2
    import os

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "1_roi_bgr.png"), roi_bgr)
        if mask is not None:
            cv2.imwrite(os.path.join(debug_dir, "1b_palm_mask.png"), mask)

    if roi_bgr.ndim == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "2_grayscale.png"), gray)

    # ===== CLAHE: local contrast boost for palm lines =====
    # Adaptive histogram equalization makes the palm creases pop out even under
    # uneven webcam lighting. clipLimit raise -> punchier but noisier;
    # tileGridSize smaller -> more local. (2.0, 8x8) is the palm-print standard.
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = _CLAHE.apply(gray)
    # ===== /CLAHE =====
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "3_clahe.png"), gray)

    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "4_resized_128.png"), gray)

    # Apply palm mask — drops background so its texture never reaches the CNN
    # and so NormSingleROI's mean/std are computed only over palm pixels.
    if mask is not None:
        mask_small = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        gray = cv2.bitwise_and(gray, gray, mask=mask_small)
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "4b_masked_128.png"), gray)

    t = torch.from_numpy(gray).float() / 255.0          # (H, W) in [0,1]
    t = t.unsqueeze(0)                                  # (1, H, W)
    # NormSingleROI: mean/std over non-zero pixels only
    flat = t.view(1, -1)
    nz = flat > 0
    if nz.any():
        vals = flat[nz]
        m = vals.mean()
        s = vals.std() + 1e-6
        flat[nz] = (vals - m) / s
    t = flat.view(1, size, size)

    if debug_dir is not None:
        # The normalized tensor is ~N(0,1) — rescale to 0..255 for viewing
        vis = t.squeeze(0).numpy()
        vmin, vmax = float(vis.min()), float(vis.max())
        vis = ((vis - vmin) / (vmax - vmin + 1e-6) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "5_normalized.png"), vis)

    return t.unsqueeze(0)                               # (1, 1, H, W)
