import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmotionResNet(nn.Module):
    """
    ResNet-18 based emotion classifier with transfer learning.

    Uses ImageNet pretrained weights and fine-tunes for emotion recognition.
    This approach typically achieves 90%+ accuracy on RAF-DB dataset.

    Architecture:
    - ResNet-18 backbone (pretrained on ImageNet)
    - Custom classification head with dropout
    - 7-class output (emotions)
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize ResNet-based emotion classifier.

        Args:
            num_classes: Number of emotion classes
            dropout_rate: Dropout rate for regularization
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers (only train classifier)
        """
        super(EmotionResNet, self).__init__()

        self.num_classes = num_classes

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features

        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[EmotionResNet] Backbone frozen, only training classifier")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Initialize classifier weights
        self._initialize_classifier()

        print(f"[EmotionResNet] Initialized with pretrained={pretrained}, "
              f"freeze_backbone={freeze_backbone}")

    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor

        Returns:
            Softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor of shape (batch, 512)
        """
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers: int = None):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("[EmotionResNet] All backbone layers unfrozen")
        else:
            # Get all layers
            layers = list(self.backbone.children())
            # Unfreeze last num_layers
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"[EmotionResNet] Last {num_layers} backbone layers unfrozen")


class EmotionEfficientNet(nn.Module):
    """
    EfficientNet-B0 based emotion classifier for even higher accuracy.
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.5,
        pretrained: bool = True
    ):
        super(EmotionEfficientNet, self).__init__()

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get the number of features
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


if __name__ == "__main__":
    # Test ResNet model
    print("Testing EmotionResNet:")
    model = EmotionResNet()

    x = torch.randn(4, 3, 100, 100)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test EfficientNet model
    print("\nTesting EmotionEfficientNet:")
    model_eff = EmotionEfficientNet()
    output_eff = model_eff(x)
    print(f"Output shape: {output_eff.shape}")
