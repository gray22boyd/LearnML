import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

# Import model from sibling file
sys.path.append(os.path.dirname(__file__))
from simple_CNN import SimpleCNN

# Load trained model (without printing)
model_path = os.path.join(os.path.dirname(__file__), "trained_cnn.pth")
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load MNIST test image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "data"),
                               train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Get one image and label
image, label = next(iter(test_loader))

# Predict label
with torch.no_grad():
    output = model(image)
    predicted = output.argmax(dim=1).item()

# Print results
print("True label:", label.item())
print("Predicted label:", predicted)

# Show image
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Input Image - Predicted: {predicted}")
plt.axis('off')
plt.show()
