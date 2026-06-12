import torch
import torch.nn as nn
import timm
from torchvision import transforms
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class SOTAEfficientNet:
    """
    SOTA EfficientNet-B0 implementation for FER-2013 using PyTorch and timm.
    """
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, model_name='efficientnet_b0', n_classes=7, pretrained=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        
        # Load pre-trained model from timm
        print(f"Loading {model_name} with pretrained={pretrained}...")
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)
        
        # Move model to device
        self.model.to(self.device)
        
        # Define transformations
        # FER-2013 is 48x48 grayscale. Pre-trained models expect 224x224 RGB.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), # Convert 1 channel to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.is_trained = False

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        return self.transform

class FERDatasetPyTorch(torch.utils.data.Dataset):
    """Custom Dataset for FER-2013 images in PyTorch."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def train_sota_model(model_wrapper, train_loader, val_loader, epochs=10, lr=1e-4):
    """
    Fine-tune the SOTA model.
    """
    print("\nStarting SOTA Model Fine-tuning (EfficientNet-B0)")
    print("=" * 60)
    
    model = model_wrapper.model
    device = model_wrapper.device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time:.2f} seconds.")
    model_wrapper.is_trained = True
    return total_train_time

def evaluate_sota_model(model_wrapper, test_loader):
    """
    Evaluate the SOTA model and calculate required metrics.
    """
    print("\nEvaluating SOTA Model Performance")
    print("=" * 60)
    
    model = model_wrapper.model
    device = model_wrapper.device
    model.eval()
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    total_inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  F1-Score:  {f1_weighted*100:.2f}%")
    print(f"  Inference time: {total_inference_time:.3f}s")
    
    return {
        'accuracy': float(accuracy),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall),
        'inference_time': float(total_inference_time),
        'confusion_matrix': cm.tolist()
    }
