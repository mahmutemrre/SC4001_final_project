"""
Model definitions for Fashion-MNIST clothing classification:
  - BaselineCNN    : standard two-block CNN baseline
  - DilatedCNN     : CNN with dilated convolutions in block 2
  - SE_DilatedCNN  : DilatedCNN + Squeeze-and-Excitation channel attention (novelty)
  - SimplViT       : lightweight Vision Transformer (patch-based)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Baseline CNN
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Standard CNN with two conv blocks."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 14x14
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 2. Dilated CNN  (novelty: dilation > 1 to widen receptive field)
# ---------------------------------------------------------------------------
class DilatedCNN(nn.Module):
    """CNN that replaces standard convolutions with dilated ones in deeper layers.

    Args:
        dilation: dilation rate for block-2 convolutions (1=standard, 2=default, 3=aggressive).
                  padding is set equal to dilation so spatial dimensions are preserved.
    """

    def __init__(self, num_classes=10, dilation=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — standard convolutions
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 14x14
            nn.Dropout2d(0.25),

            # Block 2 — dilated convolutions (padding=dilation keeps spatial size)
            nn.Conv2d(32, 64, 3, padding=dilation, dilation=dilation), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=dilation, dilation=dilation), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 3. SE-DilatedCNN  (novelty: dilated convolutions + channel attention)
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (Hu et al., 2018).

    Learns per-channel importance weights via global average pooling
    followed by a bottleneck FC → ReLU → FC → Sigmoid.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)  # avoid degenerate bottleneck
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply channel attention: pool → excite → scale."""
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)          # squeeze: (B, C)
        w = self.fc(w).view(b, c, 1, 1)      # excitation weights: (B, C, 1, 1)
        return x * w                          # scale feature maps


class SE_DilatedCNN(nn.Module):
    """DilatedCNN enhanced with Squeeze-and-Excitation channel attention.

    Inserts an SE block after each convolutional block to let the network
    learn which feature channels are most informative.

    Args:
        dilation: dilation rate for block-2 convolutions (default 2).
        reduction: SE bottleneck reduction ratio (default 16).
    """

    def __init__(self, num_classes=10, dilation=2, reduction=16):
        super().__init__()
        # Block 1 — standard convolutions + SE attention
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.se1 = SEBlock(32, reduction)
        self.pool1 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(0.25))

        # Block 2 — dilated convolutions + SE attention
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.se2 = SEBlock(64, reduction)
        self.pool2 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(0.25))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.se1(self.block1(x))
        x = self.pool1(x)
        x = self.se2(self.block2(x))
        x = self.pool2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 4. Simple Vision Transformer (patch-based)
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to embedding dim."""

    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        """Patchify, project, prepend CLS token, add positional embeddings."""
        x = self.proj(x).flatten(2).transpose(1, 2)          # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return x


class SimplViT(nn.Module):
    """Lightweight ViT for Fashion-MNIST."""

    def __init__(self, img_size=28, patch_size=4, embed_dim=128,
                 depth=6, num_heads=4, mlp_ratio=2.0, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])   # CLS token
        return self.head(x)
