import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# ==========================================
# Dataset Definition for Binary Classification
# ==========================================
class PadChestBinaryDataset(Dataset):
    """
    PadChest Dataset for binary classification (Normal vs Abnormal).
    Designed to work with master_table_binary.csv format.
    """
    
    # Binary class mapping: Normal=0, Abnormal=1
    CLASSES = ['Normal', 'Abnormal']
    CLASS_TO_IDX = {'Normal': 0, 'Abnormal': 1}
    
    def __init__(self, csv_file, img_dir, transform=None, split='test'):
        """
        Args:
            csv_file: Path to master_table_binary.csv
            img_dir: Directory containing the images
            transform: Optional transforms to apply
            split: One of 'train', 'validation', 'test'
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Load CSV
        try:
            self.data = pd.read_csv(csv_file)
        except Exception:
            self.data = pd.read_csv(csv_file + ".zip")

        # Filter by split
        if split and split.lower() != 'all' and 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        # Use label_group column for binary labels
        self.label_col = 'label_group'
        
        # Validate that labels are binary
        valid_labels = self.data[self.label_col].isin(['Normal', 'Abnormal'])
        if not valid_labels.all():
            invalid_count = (~valid_labels).sum()
            print(f"Warning: {invalid_count} samples with invalid labels will be excluded")
            self.data = self.data[valid_labels].reset_index(drop=True)
        
        # Class info for compatibility
        self.classes = self.CLASSES
        self.class_to_idx = self.CLASS_TO_IDX
        
        print(f"Loaded {len(self.data)} samples for split '{split}'")
        print(f"Class distribution: Normal={len(self.data[self.data[self.label_col] == 'Normal'])}, "
              f"Abnormal={len(self.data[self.data[self.label_col] == 'Abnormal'])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path
        img_name = self.data.iloc[idx]['ImageID']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Try adding .png extension if file not found
        if not os.path.exists(img_path):
            if not img_name.endswith('.png'):
                img_path = os.path.join(self.img_dir, img_name + '.png')
        
        # Load image or create placeholder
        if not os.path.exists(img_path):
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (224, 224), color='black')

        # Get binary label
        label_name = self.data.iloc[idx][self.label_col]
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# Backward compatibility alias
PadChestDataset = PadChestBinaryDataset