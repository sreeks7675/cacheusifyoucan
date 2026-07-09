import cv2
import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


import torch.nn as nn


class GradCAMWrapper(nn.Module):

    def __init__(self, expert):
        super().__init__()

        self.expert = expert

    def forward(self, x):

        logits, _ = self.expert(x)

        return logits



def generate_gradcam(
    model,
    image_path,
     predicted_class,
    output_path="heatmap.jpg"
):

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0)

    device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)

    # Last convolution layer of EfficientNet-B0
    target_layers = [model.backbone.blocks[-1][0]]

    wrapped_model = GradCAMWrapper(model)

    cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers
           )

    target = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=target
         )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    print(f"Heatmap saved to {output_path}")
