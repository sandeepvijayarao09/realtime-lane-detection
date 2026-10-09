"""
LaneNet: Real-time Lane Detection Network

Architecture: Encoder-Decoder CNN with skip connections
- Backbone: EfficientNet or MobileNet from torchvision
- Segmentation head: Binary lane mask prediction
- Embedding head: Instance-level lane representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsample to the skip's resolution, concatenate, refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Resize to the exact skip size so inputs need not be multiples of 32
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Slices of torchvision's `features` that end at strides 2, 4, 8, 16 and 32.
# The final 1x1 "head" conv (1280 channels) is deliberately left out.
ENCODER_STAGES = {
    'efficientnet': [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8)],
    'mobilenet': [(0, 2), (2, 4), (4, 7), (7, 14), (14, 18)],
}


def _build_backbone(backbone: str, pretrained: bool) -> nn.Module:
    """Create a torchvision backbone, optionally with ImageNet weights."""
    if backbone == 'efficientnet':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return models.efficientnet_b0(weights=weights)
    if backbone == 'mobilenet':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return models.mobilenet_v2(weights=weights)
    raise ValueError(f"Backbone must be 'efficientnet' or 'mobilenet', got {backbone}")


class LaneNet(nn.Module):
    """
    LaneNet for real-time lane detection.

    Args:
        num_classes: Number of segmentation output channels (1 = binary lane mask)
        backbone: 'efficientnet' (EfficientNet-B0) or 'mobilenet' (MobileNetV2)
        pretrained: Load ImageNet weights for the backbone (downloads on first use)
        embedding_dim: Dimension of the per-pixel instance embedding
    """

    def __init__(self, num_classes: int = 1, backbone: str = 'efficientnet',
                 pretrained: bool = True, embedding_dim: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone

        base_model = _build_backbone(backbone, pretrained)
        self._setup_encoder(base_model, backbone)

        # Read channel counts and strides off the real encoder instead of hardcoding them
        self.backbone_channels, self.backbone_strides = self._probe_encoder()

        c0, c1, c2, c3, c4 = self.backbone_channels
        self.decoder4 = DecoderBlock(c4, c3, 256)
        self.decoder3 = DecoderBlock(256, c2, 128)
        self.decoder2 = DecoderBlock(128, c1, 64)
        self.decoder1 = DecoderBlock(64, c0, 32)

        # Final layers
        self.final_conv = ConvBlock(32, 32)

        # Segmentation head (binary lane mask)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

        # Instance embedding head
        self.emb_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, embedding_dim, kernel_size=1)
        )

        self._init_weights()

    def _setup_encoder(self, base_model: nn.Module, backbone: str) -> None:
        """Split the backbone into five stages (strides 2, 4, 8, 16, 32)."""
        features = base_model.features
        stages = [nn.Sequential(*features[a:b]) for a, b in ENCODER_STAGES[backbone]]
        self.enc0, self.enc1, self.enc2, self.enc3, self.enc4 = stages

    def encoder_stages(self):
        return [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]

    @torch.no_grad()
    def _probe_encoder(self, size: int = 64):
        """Run a dummy input through the encoder to get each stage's channels and stride."""
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 3, size, size)
        channels, strides = [], []
        for stage in self.encoder_stages():
            x = stage(x)
            channels.append(x.shape[1])
            strides.append(size // x.shape[-1])
        self.train(was_training)
        return channels, strides

    def _init_weights(self) -> None:
        """Initialize decoder and head weights. The (possibly pretrained) encoder is left alone."""
        new_modules = [self.decoder4, self.decoder3, self.decoder2, self.decoder1,
                       self.final_conv, self.seg_head, self.emb_head]
        for module in new_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with:
                'seg': Segmentation logits (B, 1, H, W)
                'emb': Instance embeddings (B, embedding_dim, H, W)
        """
        input_size = x.shape[-2:]

        # Encoder with skip connections
        skip0 = self.enc0(x)  # 1/2
        skip1 = self.enc1(skip0)  # 1/4
        skip2 = self.enc2(skip1)  # 1/8
        skip3 = self.enc3(skip2)  # 1/16
        bottleneck = self.enc4(skip3)  # 1/32

        # Decoder with skip connections
        dec4 = self.decoder4(bottleneck, skip3)  # 1/16
        dec3 = self.decoder3(dec4, skip2)  # 1/8
        dec2 = self.decoder2(dec3, skip1)  # 1/4
        dec1 = self.decoder1(dec2, skip0)  # 1/2

        # Final conv and heads run at 1/2 resolution (cheap); outputs are
        # upsampled to the input size
        x = self.final_conv(dec1)

        # Heads
        seg_logits = F.interpolate(self.seg_head(x), size=input_size,
                                   mode='bilinear', align_corners=False)
        embeddings = F.interpolate(self.emb_head(x), size=input_size,
                                   mode='bilinear', align_corners=False)

        return {
            'seg': seg_logits,
            'emb': embeddings
        }

    def print_summary(self) -> None:
        """Print model summary and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"LaneNet Model Summary")
        print(f"{'='*60}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (float32)")
        print(f"Encoder channels: {self.backbone_channels}, strides: {self.backbone_strides}")
        print(f"{'='*60}\n")

        # Print layer breakdown
        print(f"{'Layer':<30} {'Parameters':<15} {'Trainable':<10}")
        print(f"{'-'*55}")
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:<30} {params:>14,} {str(trainable > 0):<10}")
        print(f"{'-'*55}\n")


def create_lanenet(num_classes: int = 1, backbone: str = 'efficientnet',
                   pretrained: bool = True, embedding_dim: int = 4) -> LaneNet:
    """
    Factory function to create LaneNet model.

    Args:
        num_classes: Number of lane instances
        backbone: Backbone architecture ('efficientnet' or 'mobilenet')
        pretrained: Load ImageNet weights for the backbone
        embedding_dim: Embedding dimension

    Returns:
        LaneNet model instance
    """
    return LaneNet(num_classes=num_classes, backbone=backbone,
                   pretrained=pretrained, embedding_dim=embedding_dim)


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_lanenet(backbone='efficientnet', pretrained=False)
    model = model.to(device)
    model.print_summary()

    # Forward pass test
    batch_size = 4
    x = torch.randn(batch_size, 3, 384, 640).to(device)
    with torch.no_grad():
        outputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {outputs['seg'].shape}")
    print(f"Embedding output shape: {outputs['emb'].shape}")
