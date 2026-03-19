#!/usr/bin/env python
# coding: utf-8

# # MNIST Digit Recognition — CNN Training
# 
# This notebook trains a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset.
# 
# **Goal**: Build a model that classifies handwritten digits (0-9) with >99% accuracy, then export it for real-time inference in our web app.

# ## 1. Setup & Imports

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set style for plots
plt.style.use('dark_background')
sns.set_theme(style='darkgrid')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ## 2. Hyperparameters

# In[2]:


# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 10

print(f'Batch Size: {BATCH_SIZE}')
print(f'Learning Rate: {LEARNING_RATE}')
print(f'Epochs: {NUM_EPOCHS}')


# ## 3. Load MNIST Dataset

# In[3]:


# Data transforms — normalize to mean=0.1307, std=0.3081 (MNIST stats)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='../data', train=True, download=True, transform=transform
)

# Download and load test data
test_dataset = datasets.MNIST(
    root='../data', train=False, download=True, transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Test samples: {len(test_dataset)}')
print(f'Image shape: {train_dataset[0][0].shape}')


# ## 4. Visualize Sample Digits

# In[4]:


# Display a grid of sample digits
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle('Sample MNIST Digits', fontsize=16, color='white')

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=10, color='#00ff88')
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[5]:


# Class distribution
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(10), [labels.count(i) for i in range(10)], 
       color=['#00ff88', '#00e5ff', '#ff6090', '#ffd740', '#b388ff',
              '#69f0ae', '#40c4ff', '#ff8a65', '#ea80fc', '#84ffff'])
ax.set_xlabel('Digit', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Training Set Class Distribution', fontsize=14, color='white')
ax.set_xticks(range(10))
plt.tight_layout()
plt.show()


# ## 5. CNN Model Architecture
# 
# ```
# Input (1x28x28)
#   │
#   ├─► Conv2D(1→32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)  →  [32x14x14]
#   │
#   ├─► Conv2D(32→64, 3x3) → BatchNorm → ReLU → MaxPool(2x2) →  [64x7x7]
#   │
#   ├─► Flatten → [3136]
#   │
#   ├─► Linear(3136→128) → ReLU → Dropout(0.5)
#   │
#   └─► Linear(128→10) → Output probabilities
# ```

# In[6]:


class MNISTNet(nn.Module):
    """CNN for MNIST digit classification."""

    def __init__(self):
        super(MNISTNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling & dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # Conv block 1: [batch, 1, 28, 28] → [batch, 32, 14, 14]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2: [batch, 32, 14, 14] → [batch, 64, 7, 7]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        # Flatten: [batch, 64, 7, 7] → [batch, 3136]
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


# Initialize model
model = MNISTNet().to(device)
print(model)
print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')


# ## 6. Training Setup

# In[7]:


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f'Loss: CrossEntropyLoss')
print(f'Optimizer: Adam (lr={LEARNING_RATE})')
print(f'Scheduler: StepLR (step=5, gamma=0.5)')


# ## 7. Training Loop

# In[ ]:


# Track metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(NUM_EPOCHS):
    # ─── Training ───
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ─── Evaluation ───
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    scheduler.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]  '
          f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  │  '
          f'Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%')

print(f'\n✅ Training complete! Best test accuracy: {max(test_accuracies):.2f}%')


# ## 8. Training Curves

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

# Loss curve
ax1.plot(epochs_range, train_losses, 'o-', color='#00ff88', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, test_losses, 's-', color='#ff6090', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Curve', fontsize=14, color='white')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2.plot(epochs_range, train_accuracies, 'o-', color='#00e5ff', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_range, test_accuracies, 's-', color='#ffd740', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Accuracy Curve', fontsize=14, color='white')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ## 9. Confusion Matrix

# In[ ]:


# Get all predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='gray')
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Confusion Matrix on Test Set', fontsize=15)
plt.tight_layout()
plt.show()

# Per-class accuracy
print('\nPer-class Accuracy:')
print('-' * 30)
for i in range(10):
    class_acc = cm[i, i] / cm[i].sum() * 100
    print(f'  Digit {i}: {class_acc:.1f}%')


# ## 10. Sample Predictions

# In[ ]:


# Visualize predictions on random test samples
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

indices = np.random.choice(len(test_dataset), 10, replace=False)

for i, ax in enumerate(axes.flat):
    image, true_label = test_dataset[indices[i]]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_label = probs.argmax()
        confidence = probs[pred_label] * 100

    ax.imshow(image.squeeze(), cmap='gray')
    color = '#00ff88' if pred_label == true_label else '#ff4444'
    ax.set_title(f'Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}', 
                 fontsize=10, color=color)
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
             fontsize=14, color='white')
plt.tight_layout()
plt.show()


# ## 11. Save Model

# In[ ]:


# Create model directory
os.makedirs('../model', exist_ok=True)

# Save model weights
model_path = '../model/mnist_cnn.pth'
torch.save(model.state_dict(), model_path)

# Verify the saved model
file_size = os.path.getsize(model_path) / 1024  # KB
print(f'✅ Model saved to: {model_path}')
print(f'   File size: {file_size:.1f} KB')

# Quick verification — reload and test
test_model = MNISTNet().to(device)
test_model.load_state_dict(torch.load(model_path, map_location=device))
test_model.eval()

with torch.no_grad():
    sample, label = test_dataset[0]
    output = test_model(sample.unsqueeze(0).to(device))
    pred = output.argmax(dim=1).item()
    print(f'\n   Verification — True: {label}, Predicted: {pred} ✓' if pred == label 
          else f'\n   Verification — True: {label}, Predicted: {pred} ✗')


# ---
# 
# ### ✅ Next Steps
# 
# The trained model (`model/mnist_cnn.pth`) is now ready to be loaded by the Flask server for real-time inference.
# 
# Run the web app:
# ```bash
# python app.py
# ```
# Then open `http://localhost:5000` in your browser to draw digits and see real-time predictions!
