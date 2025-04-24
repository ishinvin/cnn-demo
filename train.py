import os
import torch
import kagglehub
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.alexnet import AlexNet

# ----- Download Dataset from Kaggle Hub -----
data_dir = kagglehub.dataset_download("ashfakyeafi/glasses-classification-dataset")
print("------------------------- Dataset downloaded to:", data_dir)

# ----- Data augmentation for training -----
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), # AlexNet expects input images of size 224x224.
    transforms.ToTensor(), # Converts an image to a PyTorch tensor.
    transforms.Normalize( # Scaling pixel values to a consistent range, normalization helps the model converge faster.
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----- Load Data -----
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validate'), transform=val_transform)
print("------------------------- Dataset classes:", train_dataset.class_to_idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ----- Model, Loss, Optimizer -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AlexNet(num_classes=2).to(device)

# ----- Load pretrained AlexNet and modify -----
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2) # replace classifier for binary classification

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ----- TensorBoard Writer -----
writer = SummaryWriter(log_dir='runs/alexnet_experiment')

# ----- Training Loop with Early Stopping -----
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train() # set the module in training mode
    train_loss = 0
    correct_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # reset the gradients of all optimized
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()

    avg_train_loss = train_loss / len(train_dataset)
    train_accuracy = correct_train / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0
    correct_val = 0

    with torch.no_grad(): # disables gradient calculation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = correct_val / len(val_dataset)

    # ----- TensorBoard Logging -----
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # ----- Console Logging -----
    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2%} | "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2%}"
    )

    # ----- Saving best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")

# ----- Close TensorBoard -----
writer.close()

    