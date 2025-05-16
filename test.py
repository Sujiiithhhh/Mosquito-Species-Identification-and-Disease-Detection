import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

from data_utils import load_dataset
from visualize import plot_f1_scores_per_class, plot_confusion_matrix_by_genus, save_normalized_confusion_matrix

# Set paths
dataset_path = 'Images'  # Path to the test dataset
checkpoint_dir = 'checkpoints'  # Directory where the best model is saved
best_model_path = f'{checkpoint_dir}/BEST_MODEL.pth'

# Ensure the checkpoint directory exists
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
print(f"Saving plots to {checkpoint_dir}")

# Load the test dataset
_, _, _, test_loader = load_dataset(dataset_path, checkpoint_dir, batch_size=32)

# Initialize the model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(test_loader.dataset.dataset.classes))
)

# Load the best model checkpoint
model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
model.eval()

# Testing phase
device = torch.device('cpu')
y_true, y_pred = [], []
image_paths = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())
        # Corrected: Capture image paths from the original dataset
        image_paths.extend([test_loader.dataset.dataset.imgs[idx][0] for idx in test_loader.dataset.indices])

# Generate classification report
classes = test_loader.dataset.dataset.classes
print(classification_report(y_true, y_pred, target_names=classes))

# Save predictions to CSV file
csv_file_path = os.path.join(checkpoint_dir, 'predicted_species.csv')
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Path', 'Actual Species', 'Predicted Species'])
    for img_path, true_idx, pred_idx in zip(image_paths, y_true, y_pred):
        writer.writerow([img_path, classes[true_idx], classes[pred_idx]])
print(f'Predicted species saved to {csv_file_path}')

# Visualization
print("Generating F1 score plot...")
plot_f1_scores_per_class(f1_score(y_true, y_pred, average=None), classes, checkpoint_dir)

print("Generating confusion matrix by genus...")
plot_confusion_matrix_by_genus(y_true, y_pred, classes, checkpoint_dir)

print("Generating normalized confusion matrix...")
save_normalized_confusion_matrix(y_true, y_pred, classes, checkpoint_dir)

# Test save function with a simple plot
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Plot")
plt.savefig(os.path.join(checkpoint_dir, 'test_plot.png'))
plt.close()

print("Testing complete! Evaluation metrics and plots are saved.")