import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# 1. TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. LOAD DATASET
# Your Kaggle folder should have subfolders: /yes and /no (or /tumor and /healthy)
dataset = ImageFolder(root='data/', transform=transform)

# 3. SPLIT
train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

# 4. MODEL
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 5. LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 6. TRAINING LOOP
for epoch in range(10):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_set)
    print(f"Epoch {epoch+1}: loss={running_loss:.3f}, train_acc={acc:.3f}")

# 7. SAVE CHECKPOINT
torch.save(model.state_dict(), 'baseline_resnet18.pth')
print("Model saved.")