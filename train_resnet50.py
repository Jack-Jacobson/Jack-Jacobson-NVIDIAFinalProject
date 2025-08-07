#!/usr/bin/env python3
"""
ResNet50 Character Classification Training
Fine-tune a pretrained ResNet50 model on character dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

class CharacterDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Load all image paths and labels
        self.samples = []
        class_counts = {}
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            count = 0
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
                        count += 1
            class_counts[class_name] = count
        
        print(f"Found {len(self.samples)} images in {len(class_names)} classes")
        
        # Print class distribution
        print("📊 Class distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} images")
        
        # Calculate class weights for imbalanced dataset
        total_samples = len(self.samples)
        self.class_weights = torch.zeros(len(class_names))
        for class_name, count in class_counts.items():
            if count > 0:
                self.class_weights[self.class_to_idx[class_name]] = total_samples / (len(class_names) * count)
            else:
                self.class_weights[self.class_to_idx[class_name]] = 1.0
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNet50Trainer:
    def __init__(self, num_classes, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Initialize model
        self.model = models.resnet50(pretrained=True)
        
        # Modify final layer for our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)
        
        # Loss function and optimizer (with class weights for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"✅ ResNet50 model initialized on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50):
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        
        best_val_acc = 0.0
        best_model_path = "resnet50_best.pth"
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': self.class_names
                }, best_model_path)
                print(f"💾 New best model saved! Val Acc: {best_val_acc:.2f}%")
        
        print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_model_path
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('resnet50_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Training curves saved as 'resnet50_training_curves.png'")

def load_config():
    """Load dataset configuration"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'Database', 'data.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_data_transforms():
    """Define data transformations"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config = load_config()
    class_names = config['names']
    num_classes = len(class_names)
    
    print(f"🎯 Training ResNet50 for {num_classes} character classes")
    print(f"📝 Classes: {class_names}")
    
    # Get data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dataset = CharacterDataset(
        data_dir=os.path.join(script_dir, 'Database', 'train'),
        class_names=class_names,
        transform=train_transform
    )
    
    val_dataset = CharacterDataset(
        data_dir=os.path.join(script_dir, 'Database', 'val'),
        class_names=class_names,
        transform=val_transform
    )
    
    # Get class weights from training dataset
    class_weights = train_dataset.class_weights
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Batch size: {batch_size}")
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ResNet50Trainer(num_classes=num_classes, class_names=class_names, device=device)
    
    # Apply class weights to loss function
    trainer.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"📊 Applied class weights to handle imbalanced dataset")
    
    # Train the model
    best_model_path = trainer.train(train_loader, val_loader, num_epochs=30)
    
    # Plot training curves
    trainer.plot_training_history()
    
    print(f"\n✅ Training completed!")
    print(f"💾 Best model saved as: {best_model_path}")
    print(f"📊 Training curves saved as: resnet50_training_curves.png")
    
    # Save final model
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes
    }, 'resnet50_final.pth')
    print(f"💾 Final model saved as: resnet50_final.pth")

if __name__ == '__main__':
    main()
