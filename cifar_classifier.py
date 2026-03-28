import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# --- Settings ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load CIFAR-10 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Downloading CIFAR-10 (this only happens once)...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- Define the CNN ---
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # Convolutional layers — extract visual features
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 3 color channels in, 32 feature maps out
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # 32 in, 64 out
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 64 in, 128 out
        self.pool = nn.MaxPool2d(2, 2)                  # shrinks image by half each time

        # Fully connected layers — make the final decision
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classes

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(torch.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = self.pool(torch.relu(self.conv3(x)))   # 8x8 -> 4x4

        x = x.view(-1, 128 * 4 * 4)  # flatten into a 1D vector
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Set up model, loss, optimizer ---
model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
print("\nTraining...")
for epoch in range(10):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"  Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

# --- Test accuracy ---
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nAccuracy on 10,000 test images: {100 * correct / total:.1f}%")

# --- Show some predictions ---
print("\nSample predictions:")
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

for i in range(10):
    actual = classes[labels[i]]
    guess = classes[predicted[i]]
    status = "✓" if actual == guess else "✗"
    print(f"  {status} Actual: {actual:>6}  |  Predicted: {guess}")