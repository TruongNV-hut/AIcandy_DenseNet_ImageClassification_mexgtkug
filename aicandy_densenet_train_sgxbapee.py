"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms
from aicandy_model_src_bpvbytql.aicandy_model_densenet_ekktouop import DenseNet
import os
import json

cudnn.benchmark = True
cudnn.deterministic = True

# Command:
# python aicandy_densenet_train_sgxbapee.py --data_dir ../dataset --num_epochs 10 --batch_size 16 --model_path aicandy_model_out_ddmalncc/aicandy_model_pth_silsegko.pth


def train(data_dir, num_epochs, batch_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with: ", device)

    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),      # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Save class labels
    with open('label.txt', 'w') as f:
        for idx, class_name in enumerate(dataset.classes):
            f.write(f'{idx}: {class_name}\n')

    num_classes = len(dataset.classes)
    model = DenseNet(num_blocks=[6, 12, 24, 16], growth_rate=12, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * running_corrects.double() / len(train_dataset)

        # Đánh giá trên validation set
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_dataset)
        val_acc = 100 * val_corrects.double() / len(val_dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data folder')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model and labels')

    args = parser.parse_args()

    train(args.data_dir, args.num_epochs, args.batch_size, args.model_path)
