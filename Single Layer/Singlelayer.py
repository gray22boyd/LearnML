import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔹 Define a simple neural network model
class SimplePerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        print("🧠 Initializing SimplePerceptron...")
        print(" - Input features: 8 (e.g., glucose, insulin, BMI, etc.)")
        print(" - Hidden layer: 6 neurons")
        print(" - Output layer: 1 neuron (yes/no prediction)")

        self.fc1 = nn.Linear(8, 6)      # First layer: 8 → 6
        self.fc2 = nn.Linear(6, 1)      # Second layer: 6 → 1
        self.sigmoid = nn.Sigmoid()     # Use sigmoid to output probability

    def forward(self, x):
        x = self.fc1(x)
        print(f"🔁 Output after fc1 (before ReLU): {x[0].detach().numpy()}")
        x = torch.relu(x)               # Activation function
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 🔹 Train the model
def train_model():
    print("📥 Step 1: Loading dataset...")
    df = pd.read_csv("pima_diabetes.csv", header=None)
    print(f"✔️ Dataset loaded with shape: {df.shape} (rows, columns)")

    X = df.iloc[:, :-1].values   # Features: all columns except last
    y = df.iloc[:, -1].values    # Target: last column (0 or 1)

    print("\n📊 Sample raw input:")
    print(f" - Features: {X[0]}")
    print(f" - Label: {y[0]}")

    print("\n🔄 Step 2: Normalizing input features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f" - Normalized first input row: {X[0]}")

    print("\n✂️ Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" - Training samples: {len(X_train)}")
    print(f" - Testing samples: {len(X_test)}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    print("\n🧠 Step 4: Creating model, loss function, and optimizer...")
    model = SimplePerceptron()
    loss_fn = nn.BCELoss()  # Binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\n🚀 Step 5: Training the model...")
    for epoch in range(1, 51):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"📈 Epoch {epoch:2d}: Loss = {loss.item():.4f}")

    print("\n🧪 Step 6: Evaluating model...")
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        pred_labels = (preds >= 0.5).float()
        accuracy = (pred_labels == y_test).float().mean()

        print("\n🔍 Sample predictions:")
        for i in range(5):
            print(f" - Predicted: {int(pred_labels[i].item())}, Actual: {int(y_test[i].item())}")

        print(f"\n✅ Final Test Accuracy: {accuracy:.4f}")

    # ✅ Save the trained model for visualize.py
    torch.save(model.state_dict(), "trained_model.pth")
    print("✅ Model saved as 'trained_model.pth'")

    return model

# 🔹 Reusable function to load trained model
def get_model():
    model = SimplePerceptron()
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()
    return model

# 🔹 Run training if file is executed directly
if __name__ == "__main__":
    train_model()
