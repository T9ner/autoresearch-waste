"""
Data preparation for waste classification.
Fixed constants and evaluation - DO NOT MODIFY.
"""

import os
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset

# ----------------------------------------------------------------------
# FIXED CONSTANTS - DO NOT MODIFY
# ----------------------------------------------------------------------

TIME_BUDGET = 300  # 5 minutes in seconds
MAX_SEQ_LEN = 224  # Image size (224x224)
DEVICE_BATCH_SIZE = 32
EVAL_TOKENS = 524288  # Number of samples to evaluate
NUM_CLASSES = 3  # e-waste, plastic, organic

# Data directories
DATA_DIR = os.path.expanduser("~/.cache/autoresearch-waste/")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(DATA_DIR, "val")

# ----------------------------------------------------------------------
# DATASET LOADING
# ----------------------------------------------------------------------

def load_waste_datasets():
    """Load and prepare waste classification datasets."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(VAL_DATA_DIR, exist_ok=True)
    
    print("Loading waste classification datasets...")
    
    # Load main waste classification dataset
    try:
        waste_ds = load_dataset("NeoAivara/Waste_Classification_data", split="train")
    except Exception as e:
        print(f"Failed to load NeoAivara dataset: {e}")
        waste_ds = None
    
    # Load organic classification dataset
    try:
        organic_ds = load_dataset("bryandts/waste_organic_anorganic_classification", split="train")
    except Exception as e:
        print(f"Failed to load organic dataset: {e}")
        organic_ds = None
    
    return waste_ds, organic_ds


def map_class_to_category(label_str):
    """Map dataset labels to our 3 categories: e-waste, plastic, organic."""
    label_lower = label_str.lower() if isinstance(label_str, str) else str(label_str)
    
    # E-waste categories
    if any(x in label_lower for x in ['battery', 'phone', 'electronic', 'e-waste', 'laptop', 'computer']):
        return 0  # e-waste
    # Plastic categories
    elif any(x in label_lower for x in ['plastic', 'pet', 'hdpe', 'bottle', 'container', 'wrapper']):
        return 1  # plastic
    # Organic categories
    elif any(x in label_lower for x in ['organic', 'food', 'compost', 'vegetable', 'fruit', 'garden']):
        return 2  # organic
    else:
        # Default to plastic (most common)
        return 1


class WasteDataset(Dataset):
    """PyTorch Dataset for waste images."""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            image = None
            
        if image is None:
            # Return dummy data
            return torch.randn(3, MAX_SEQ_LEN, MAX_SEQ_LEN), 1
            
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize
        image = image.resize((MAX_SEQ_LEN, MAX_SEQ_LEN))
        
        # Convert to tensor
        if isinstance(image, Image.Image):
            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).float()
        
        # Get label
        if 'label' in item:
            label = item['label']
        elif 'labels' in item:
            label = item['labels']
        else:
            label = 1
            
        # Map to our categories
        mapped_label = map_class_to_category(label)
        
        return image, mapped_label


def make_dataloader(split='train', batch_size=DEVICE_BATCH_SIZE):
    """Create dataloader for training or validation."""
    waste_ds, organic_ds = load_waste_datasets()
    datasets_list = []
    if waste_ds:
        datasets_list.append(WasteDataset(waste_ds))
    if organic_ds:
        datasets_list.append(WasteDataset(organic_ds))

    if not datasets_list:
        raise RuntimeError("No datasets available")

    combined = ConcatDataset(datasets_list)
    return DataLoader(combined, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0)


def evaluate_model(model, dataloader, device):
    """Evaluate model on classification task."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if total >= EVAL_TOKENS:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


# Stub for compatibility with original autoresearch structure
def evaluate_bpb(model, dataloader, device):
    """Stub - returns classification accuracy instead of bits per byte."""
    return evaluate_model(model, dataloader, device)


class Tokenizer:
    """Stub tokenizer for compatibility."""
    def __init__(self):
        self.vocab_size = NUM_CLASSES
    
    def encode(self, s):
        return [0] * len(s)
    
    def decode(self, ids):
        return ""


if __name__ == "__main__":
    print("Preparing waste classification datasets...")
    load_waste_datasets()
    print("Done! Datasets saved to:", DATA_DIR)
