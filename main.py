import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm  
import itertools
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

from train_eval import train_model
from visualize import save_plots, plot_f1_scores_per_class, plot_confusion_matrix_by_genus, save_normalized_confusion_matrix

# Define paths and transformation
dataset_path = 'Images'
checkpoint_dir = 'checkpoints'

# Enhanced data augmentation and regularization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and split dataset with explicit shuffling
full_dataset = ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# DataLoader with proper shuffling
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

learning_rates = [0.0001, 0.0005]
batch_sizes = [16, 32]
optimizers = ['SGD', 'Adam']

best_loss = float('inf')
best_model = None

for lr, bs, opt in itertools.product(learning_rates, batch_sizes, optimizers):
    model_dir = os.path.join(checkpoint_dir, f"lr_{lr}_bs{bs}_opt{opt}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Training with lr={lr}, batch_size={bs}, optimizer={opt}")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.7),  # Increased dropout to avoid overfitting
        nn.Linear(num_features, num_classes)
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) if opt == 'Adam' else optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)

    model, train_loss, val_loss, train_accuracy, val_accuracy, f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_classes, lr, bs, opt, checkpoint_dir, epochs=25, patience=7
    )

    if val_loss[-1] < best_loss:
        best_loss = val_loss[-1]
        best_model = model

    save_plots(train_loss, val_loss, train_accuracy, val_accuracy, f1_scores, model_dir, prefix="")

best_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

print(classification_report(y_true, y_pred, target_names=full_dataset.classes))
plot_f1_scores_per_class(f1_score(y_true, y_pred, average=None), full_dataset.classes, checkpoint_dir)
plot_confusion_matrix_by_genus(y_true, y_pred, full_dataset.classes, checkpoint_dir)
save_normalized_confusion_matrix(y_true, y_pred, full_dataset.classes, checkpoint_dir)

torch.save(best_model.state_dict(), f'{checkpoint_dir}/BEST_MODEL.pth')
