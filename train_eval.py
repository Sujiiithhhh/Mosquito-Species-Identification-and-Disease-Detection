import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score
import fnmatch

def save_checkpoint(state, checkpoint_dir, learning_rate, batch_size, optimizer_choice, epoch, filename=None):
    if filename is None:
        filename = f"checkpoint_lr_{learning_rate}_bs_{batch_size}_opt_{optimizer_choice}_epoch_{epoch}.pth.tar"
    model_dir = os.path.join(checkpoint_dir, f"lr_{learning_rate}_bs_{batch_size}_opt_{optimizer_choice}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = os.path.join(model_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_classes, learning_rate, batch_size, optimizer_choice, checkpoint_dir, epochs=15, patience=10, start_epoch=0, best_val_loss=float('inf')):
    patience_counter = 0
    epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy, epoch_f1_scores = [], [], [], [], []

    for epoch in range(start_epoch, epochs):
        device = torch.device("cpu")
        model = model.to(device)
        model.train()

        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        epoch_train_loss.append(train_loss / len(train_loader))
        epoch_train_accuracy.append(train_accuracy)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = 100 * val_correct / val_total
        epoch_val_loss.append(val_loss / len(val_loader))
        epoch_val_accuracy.append(val_accuracy)
        epoch_f1_scores.append(f1_score(all_labels, all_predictions, average='weighted'))

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_accuracy': epoch_train_accuracy,
                'val_accuracy': epoch_val_accuracy,
                'f1_scores': epoch_f1_scores,
                'best_val_loss': best_val_loss,
            }, checkpoint_dir, learning_rate, batch_size, optimizer_choice, epoch)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement")
                break

    # Save logs for visualization
    torch.save(epoch_train_loss, os.path.join(checkpoint_dir, "train_loss.pth"))
    torch.save(epoch_val_loss, os.path.join(checkpoint_dir, "val_loss.pth"))
    torch.save(epoch_train_accuracy, os.path.join(checkpoint_dir, "train_accuracy.pth"))
    torch.save(epoch_val_accuracy, os.path.join(checkpoint_dir, "val_accuracy.pth"))

    return model, epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy, epoch_f1_scores
