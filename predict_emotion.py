import sys
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from PIL import Image

# --- Constants ---
HERE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(HERE, "emotion_resnet18_best.pth")
NUM_CLASSES = 7

# ImageFolder assigns classes alphabetically:
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# --- Model (identical definition to emotion_classifier.py) ---
def build_model():
    model = models.resnet18(weights=None)  # weights loaded from checkpoint
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model

# --- Inference transform (same as val transform in emotion_classifier.py) ---
def build_inference_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# --- Entry point ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_emotion.py <path_to_face_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check model checkpoint exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model checkpoint not found at '{MODEL_PATH}'")
        print("Train the model first by running:  python emotion_classifier.py")
        sys.exit(1)

    # Check image exists
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    model = build_model().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint — epoch {checkpoint['epoch']}, val acc: {checkpoint['val_acc']:.1f}%")

    # Load and preprocess image
    transform = build_inference_transform()
    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Run inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    # Print results
    print(f"\nPredicted emotion: {EMOTIONS[pred]}\n")
    print("Confidence scores:")
    for i, (name, p) in enumerate(zip(EMOTIONS, probs)):
        bar = "#" * int(p.item() * 40)
        marker = " <--" if i == pred else ""
        print(f"  {name:<10} {p.item()*100:5.1f}%  {bar}{marker}")
