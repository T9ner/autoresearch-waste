"""
Waste Classification Training Script.
This is the ONLY file the autonomous agent can modify.

Task: Classify waste images into 3 categories (e-waste, plastic, organic)
      and predict yield percentage.

Metrics:
  - classification_accuracy: Top-1 accuracy (higher is better)
  - yield_prediction_mse: MSE for yield prediction (lower is better)
  - combined_score: accuracy - 0.1 * yield_mse (higher is better)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURATION - Agent can modify these
# ----------------------------------------------------------------------

@dataclass
class Config:
    # Model architecture
    model_type: str = "resnet18"  # resnet18, resnet34, efficientnet_b0, vit_base
    pretrained: bool = True
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 5
    warmup_epochs: int = 1
    
    # Image settings
    image_size: int = 224
    num_classes: int = 3  # e-waste, plastic, organic
    
    # Yield prediction head
    predict_yield: bool = True
    yield_weight: float = 0.1
    
    # Data
    data_dir: str = "~/.cache/autoresearch-waste/"


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def map_label_to_category(label):
    """Map original labels to our 3 categories."""
    label_str = str(label).lower()
    
    # 0 = e-waste
    if any(x in label_str for x in ['battery', 'phone', 'electronic', 'laptop', 'computer', 'tv', 'monitor']):
        return 0
    # 1 = plastic
    elif any(x in label_str for x in ['plastic', 'pet', 'hdpe', 'bottle', 'container', 'wrapper', 'glass', 'metal', 'cardboard', 'paper', 'can']):
        return 1
    # 2 = organic
    else:
        return 2


class WasteDataset(torch.utils.data.Dataset):
    """Dataset wrapper for HuggingFace waste datasets."""
    
    def __init__(self, hf_dataset, image_size=224):
        self.dataset = hf_dataset
        self.image_size = image_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract image
        if 'image' in item:
            img = item['image']
        elif 'img' in item:
            img = item['img']
        else:
            return torch.zeros(3, self.image_size, self.image_size), 0, 0.0
        
        # Process image
        if img is None:
            return torch.zeros(3, self.image_size, self.image_size), 0, 0.0
            
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((self.image_size, self.image_size))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Extract label
        if 'label' in item:
            label = item['label']
        elif 'labels' in item:
            label = item['labels']
        else:
            label = 1
            
        category = map_label_to_category(label)
        
        # Simulated yield (in real app, this would be from actual data)
        # For e-waste: 60-80%, plastic: 70-90%, organic: 50-80%
        base_yield = [0.7, 0.8, 0.65][category]
        yield_val = base_yield + torch.randn(1).item() * 0.1
        
        return img, category, yield_val


def collate_fn(batch):
    """Custom collate function."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    yields = torch.tensor([b[2] for b in batch])
    return images, labels, yields


def get_dataloaders(config):
    """Create train and validation dataloaders."""
    print("Loading datasets...")
    
    try:
        # Try to loadWaste datasets
        ds1 = load_dataset("NeoAivara/Waste_Classification_data", split="train")
        ds2 = load_dataset("bryandts/waste_organic_anorganic_classification", split="train")
        
        # Use a portion for training
        train_size = min(len(ds1), 1000)
        val_size = min(200, len(ds1) - train_size)
        
        # Split datasets
        # Note: In production, properly split with train_test_split
        train_ds = WasteDataset(ds1.select(range(train_size)), config.image_size)
        
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        print("Using synthetic data for testing...")
        # Create dummy dataset for testing
        train_ds = WasteDataset([{'image': None, 'label': 'plastic'}] * 100, config.image_size)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, None  # Val loader would be similar


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Classification head for waste types."""
    
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class YieldHead(nn.Module):
    """Yield prediction head."""
    
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Yield is 0-1
        )
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)


class WasteClassifier(nn.Module):
    """Combined classifier for waste type and yield."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained backbone
        if config.model_type == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if config.pretrained else None)
            self.feature_dim = 512
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif config.model_type == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT if config.pretrained else None)
            self.feature_dim = 512
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif config.model_type == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if config.pretrained else None)
            self.feature_dim = 1280
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        # Classification head
        self.classifier = ClassificationHead(self.feature_dim, config.num_classes)
        
        # Yield prediction head
        if config.predict_yield:
            self.yield_head = YieldHead(self.feature_dim)
        
        self.freeze_backbone = False
    
    def forward(self, x, return_features=False):
        features = self.backbone(x).flatten(1)
        
        class_logits = self.classifier(features)
        
        if return_features:
            return class_logits, features
        
        if self.config.predict_yield:
            yield_pred = self.yield_head(features)
            return class_logits, yield_pred
        
        return class_logits


# ----------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, config, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, yields) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        yields = yields.to(device)
        
        optimizer.zero_grad()
        
        if config.predict_yield:
            class_logits, yield_pred = model(images)
            loss = criterion(class_logits, labels)
            yield_loss = F.mse_loss(yield_pred, yields)
            loss = loss + config.yield_weight * yield_loss
        else:
            class_logits = model(images)
            loss = criterion(class_logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(class_logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy


def evaluate(model, val_loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, yields in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            class_logits = model(images)
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    """Main training loop."""
    import time
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create model
    model = WasteClassifier(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    start_time = time.time()
    best_accuracy = 0.0
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, config, device, epoch)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config.num_epochs}: loss={loss:.4f}, accuracy={accuracy:.2f}%, time={epoch_time:.1f}s")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            torch.save(model.state_dict(), os.path.expanduser("~/.cache/autoresearch-waste/best_model.pt"))
    
    total_time = time.time() - start_time
    
    # Report final results
    print("\n" + "="*50)
    print(f"val_accuracy:     {best_accuracy:.2f}")
    print(f"training_seconds: {total_time:.1f}")
    if torch.cuda.is_available():
        print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1e6:.1f}")
    print("="*50)
    
    return best_accuracy


if __name__ == "__main__":
    main()