import sys
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models import ResNet18_Weights

# --- Constants ---
HERE        = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(HERE, "fer-2013", "train")
TEST_DIR    = os.path.join(HERE, "fer-2013", "test")
MODEL_PATH  = os.path.join(HERE, "emotion_resnet18_best.pth")
NUM_EPOCHS  = 15
BATCH_SIZE  = 64
VAL_SPLIT   = 0.1      # 10% of train used for validation
NUM_CLASSES = 7

# ImageFolder assigns classes alphabetically:
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# --- Data directory check ---
if not os.path.exists(TRAIN_DIR):
    print(f"Error: training data not found at '{TRAIN_DIR}'")
    print("Expected structure:")
    print("  fer-2013/train/angry/, fer-2013/train/happy/, ...")
    sys.exit(1)

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Transforms ---
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # grayscale -> 3-channel
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# --- Datasets ---
# Load train folder twice (different transforms) so val subset gets val_transform
full_train_t = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_train_v = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

n_total = len(full_train_t)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

# Same random split indices applied to both transform variants
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)

train_dataset = Subset(full_train_t, list(train_indices))
val_dataset   = Subset(full_train_v, list(val_indices))

print(f"  Train: {len(train_dataset)} images")
print(f"  Val:   {len(val_dataset)} images")
print(f"  Test:  {len(test_dataset)} images")
print(f"  Classes: {full_train_t.classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Class weights to handle imbalance (Disgust is ~1% of data) ---
label_counts = Counter(full_train_t.targets)
total = sum(label_counts.values())
weights = [total / (NUM_CLASSES * label_counts[i]) for i in range(NUM_CLASSES)]
class_weights = torch.FloatTensor(weights).to(device)

# --- Model ---
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Two param groups: lower LR for pretrained backbone, higher LR for fresh head
backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc")]
optimizer = optim.Adam([
    {"params": backbone_params,          "lr": 1e-4},
    {"params": model.fc.parameters(),    "lr": 1e-3},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --- Train ---
print("\nTraining...")
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Training pass
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # Validation pass
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss    += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc  = 100 * val_correct / val_total

    scheduler.step()

    print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}  "
          f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  "
          f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_acc":          val_acc,
        }, MODEL_PATH)
        print(f"    -> Saved new best checkpoint (val acc: {val_acc:.1f}%)")

# --- Final test evaluation ---
print(f"\nLoading best checkpoint (val acc: {best_val_acc:.1f}%)...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total   += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * test_correct / test_total:.1f}%")
print(f"\nCheckpoint saved to: {MODEL_PATH}")
print("Run predictions with:  python predict_emotion.py <path_to_face_image>")
