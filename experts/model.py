import torch
import torch.nn as nn
import timm


class ExpertModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained EfficientNet-B0 without classifier
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(1280, 2)

    def forward(self, x):

        features = self.backbone(x)

        x = self.dropout(features)

        logits = self.classifier(x)

        return logits, features