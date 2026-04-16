"""
Model definitions:
  - BaselineCNN   : simple CNN baseline
  - DilatedCNN    : CNN with dilated convolutions (novelty component)
  - SimplViT      : lightweight Vision Transformer
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
# 3. Simple Vision Transformer (patch-based)
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
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
