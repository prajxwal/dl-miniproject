import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
import os
import random
from PIL import Image, ImageDraw, ImageFilter
import time

# ─── Settings ───
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 5      # 5 epochs is enough to converge for 11 classes
NUM_CLASSES = 11    # 10 digits + 1 junk class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🚀 Starting Training with Junk Class on {device}')

# ─── Custom Junk Dataset ───
class JunkDataset(Dataset):
    def __init__(self, size, transform=None):
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Create a black 28x28 canvas
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # Add a random number of random shapes to simulate scribbles
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            shape_type = random.choice(['line', 'ellipse', 'rectangle', 'arc'])
            x1 = random.randint(0, 20)
            y1 = random.randint(0, 20)
            x2 = random.randint(x1 + 2, 28)
            y2 = random.randint(y1 + 2, 28)
            width = random.randint(1, 4)
            color = random.randint(100, 255)
            
            if shape_type == 'line':
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
            elif shape_type == 'ellipse':
                draw.ellipse([x1, y1, x2, y2], outline=color, width=width)
            elif shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            elif shape_type == 'arc':
                draw.arc([x1, y1, x2, y2], start=0, end=random.randint(90, 360), fill=color, width=width)
                
        # Blur some images to simulate soft handwriting edges
        if random.random() > 0.5:
             img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

        if self.transform:
            img = self.transform(img)
            
        return img, 10  # Label 10 = Junk

# ─── Data Preparation ───
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Standard MNIST
mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

# Synthesize Junk Class Data (~10% of total dataset)
# 60,000 MNIST training samples -> ~6,000 Junk training samples
junk_train = JunkDataset(6000, transform=transform)
junk_test = JunkDataset(1000, transform=transform)

train_dataset = ConcatDataset([mnist_train, junk_train])
test_dataset = ConcatDataset([mnist_test, junk_test])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ─── Model ───
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ─── Training Loop ───
for epoch in range(NUM_EPOCHS):
    start = time.time()
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    
    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    scheduler.step()
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} ({time.time()-start:.1f}s) | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

# ─── Save ───
os.makedirs('../model', exist_ok=True)
torch.save(model.state_dict(), '../model/mnist_cnn.pth')
print('✅ Saved to ../model/mnist_cnn.pth')
