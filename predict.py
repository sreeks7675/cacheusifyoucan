import os
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from experts.model import ExpertModel
from consensus.model import ConsensusModel
from gradcam import generate_gradcam

# Create output directory
os.makedirs("output", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_expert(checkpoint_path):
    model = ExpertModel().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


def load_consensus(checkpoint_path):
    model = ConsensusModel().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


# Load trained models
gan_expert = load_expert("checkpoint/gan_expert.pth")

compression_expert = load_expert(
    "checkpoint/compression_expert.pth"
)

diffusion_expert = load_expert(
    "checkpoint/diffusion_expert.pth"
)

consensus = load_consensus(
    "checkpoint/consensus_head.pth"
)


def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        gan_logits, gan_features = gan_expert(image)
        comp_logits, comp_features = compression_expert(image)
        diff_logits, diff_features = diffusion_expert(image)

        gan_probs = torch.softmax(gan_logits, dim=1)
        comp_probs = torch.softmax(comp_logits, dim=1)
        diff_probs = torch.softmax(diff_logits, dim=1)

        consensus_features = torch.cat([
            gan_features,
            comp_features,
            diff_features,
            gan_probs,
            comp_probs,
            diff_probs
        ], dim=1)

        logits = consensus(consensus_features)
        probabilities = F.softmax(logits, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)
        prediction = prediction.item()

        if prediction == 1:
            expert_scores = {
                "GAN": gan_probs[0][1].item(),
                "Compression": comp_probs[0][1].item(),
                "Diffusion": diff_probs[0][1].item()
            }
        else:
            expert_scores = {
                "GAN": gan_probs[0][0].item(),
                "Compression": comp_probs[0][0].item(),
                "Diffusion": diff_probs[0][0].item()
            }

    best_expert = max(expert_scores, key=expert_scores.get)

    # Enable gradients only for Grad-CAM
    with torch.enable_grad():

        if best_expert == "GAN":
            generate_gradcam(
                gan_expert,
                image_path,
                prediction,
                "output/heatmap.jpg"
            )

        elif best_expert == "Compression":
            generate_gradcam(
                compression_expert,
                image_path,
                prediction,
                "output/heatmap.jpg"
            )

        else:
            generate_gradcam(
                diffusion_expert,
                image_path,
                prediction,
                "output/heatmap.jpg"
            )

    label = "REAL" if prediction == 0 else "FAKE"

    return {
        "prediction": label,
        "confidence": confidence.item() * 100,
        "selected_expert": best_expert,
        "gan_probability": gan_probs[0][prediction].item() * 100,
        "compression_probability": comp_probs[0][prediction].item() * 100,
        "diffusion_probability": diff_probs[0][prediction].item() * 100,
        "selected_expert_confidence": expert_scores[best_expert] * 100,
        "heatmap": "output/heatmap.jpg"
    }


if __name__ == "__main__":

    result = predict("images1.jpeg")

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2f}%")
    print(f"Expert     : {result['selected_expert']}")
    print(f"Heatmap    : {result['heatmap']}")

    print("\nExpert Opinions")
    print("---------------------")
    print(f"GAN         : {result['gan_probability']:.2f}%")
    print(f"Compression : {result['compression_probability']:.2f}%")
    print(f"Diffusion   : {result['diffusion_probability']:.2f}%")
