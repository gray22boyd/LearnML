import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        print("Initializing CNN model...")
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print("After maxpool1:", x.shape)

        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print("After maxpool2:", x.shape)

        x = x.view(-1, 320)
        print("After flattening:", x.shape)
        x = F.relu(self.fc1(x))
        print("After fc1:", x.shape)
        x = self.fc2(x)
        print("After fc2 (raw scores):", x.shape)
        return x

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

# Evaluation
def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n")

# Train the model
for epoch in range(1, 4):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)

# Save the model weights
pth_path = os.path.join(os.path.dirname(__file__), "trained_cnn.pth")
torch.save(model.state_dict(), pth_path)
print(f"Model saved to {pth_path}")

# Export the model to ONNX (for Netron)
onnx_path = os.path.join(os.path.dirname(__file__), "cnn_model.onnx")
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)
print(f"ONNX model exported to {onnx_path}")
