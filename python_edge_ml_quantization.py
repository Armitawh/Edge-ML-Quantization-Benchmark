import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_model(model, loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def evaluate_speed(model, loader):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for images, _ in loader:
            _ = model(images)
    return time.time() - start


model = CNNModel()
loss_history = train_model(model, train_loader, epochs=3)

orig_acc = evaluate_accuracy(model, test_loader)
orig_time = evaluate_speed(model, test_loader)

torch.save(model.state_dict(), "original_model.pth")
orig_size = os.path.getsize("original_model.pth") / 1024


quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

quant_acc = evaluate_accuracy(quantized_model, test_loader)
quant_time = evaluate_speed(quantized_model, test_loader)

torch.save(quantized_model.state_dict(), "quantized_model.pth")
quant_size = os.path.getsize("quantized_model.pth") / 1024


print("\n--- RESULTS ---")
print(f"Original Model  | Size: {orig_size:.2f} KB | Acc: {orig_acc:.2f}% | Time: {orig_time:.2f}s")
print(f"Quantized Model | Size: {quant_size:.2f} KB | Acc: {quant_acc:.2f}% | Time: {quant_time:.2f}s")


plt.figure()
plt.plot(loss_history, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


plt.figure()
plt.bar(["Original", "Quantized"], [orig_size, quant_size])
plt.title("Model Size Comparison (KB)")
plt.ylabel("Size (KB)")
plt.show()


plt.figure()
plt.bar(["Original", "Quantized"], [orig_time, quant_time])
plt.title("Inference Time Comparison (seconds)")
plt.ylabel("Time (s)")
plt.show()

plt.figure(figsize=(14, 4))

# Loss curve
plt.subplot(1, 3, 1)
plt.plot(loss_history, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Model size
plt.subplot(1, 3, 2)
plt.bar(["FP32", "INT8"], [orig_size, quant_size])
plt.title("Model Size (KB)")
plt.ylabel("KB")

# Inference time
plt.subplot(1, 3, 3)
plt.bar(["FP32", "INT8"], [orig_time, quant_time])
plt.title("Inference Time (s)")
plt.ylabel("Seconds")

plt.suptitle("Edge ML Quantization: Accuracy–Size–Latency Trade-offs", fontsize=14)
plt.tight_layout()
plt.show()
