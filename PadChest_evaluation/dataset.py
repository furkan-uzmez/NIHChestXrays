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
    PadChest Dataset for binary classification (PA vs AP).
    Designed to work with padchest_merged.csv format.
    """
    
    # Binary class mapping: ap=0, pa=1
    CLASSES = ['AP', 'PA']
    CLASS_TO_IDX = {'AP': 0, 'PA': 1}
    
    def __init__(self, csv_file, img_dir, transform=None, split='all'):
        """
        Args:
            csv_file: Path to padchest_merged.csv
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
        # Use Projection column for binary labels (normalized to ap/pa)
        self.label_col = 'Projection'
        
        # Validate that labels are ap or pa
        valid_labels = self.data[self.label_col].isin(['PA', 'AP'])
        print(f"Valid labels: {valid_labels.sum()}")
        if not valid_labels.all():
            invalid_count = (~valid_labels).sum()
            print(f"Warning: {invalid_count} samples with invalid labels will be excluded")
            self.data = self.data[valid_labels].reset_index(drop=True)
        
        # Class info for compatibility
        self.classes = self.CLASSES
        self.class_to_idx = self.CLASS_TO_IDX
        
        print(f"Loaded {len(self.data)} samples for split '{split}'")
        print(f"Class distribution: PA={len(self.data[self.data[self.label_col] == 'PA'])}, "
              f"AP={len(self.data[self.data[self.label_col] == 'AP'])}")

        # Build full image paths
        if 'ImageID' not in self.data.columns:
            raise ValueError("CSV does not contain 'ImageID' column")

        self.data['image_path'] = self.data['ImageID'].apply(
            lambda x: os.path.join(self.img_dir, x)
        )

        # Optional: drop missing files (recommended for research)
        exists_mask = self.data['image_path'].apply(os.path.exists)
        missing = (~exists_mask).sum()
        if missing > 0:
            print(f"Warning: {missing} missing images will be dropped")

        self.data = self.data[exists_mask].reset_index(drop=True)
      

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        
        # Etiketleri TERS ÇEVİRMİYORUZ (Hile yok!)
        # Orijinal mantık: AP=0, PA=1
        label_str = row[self.label_col]
        label = self.class_to_idx[label_str]

        # 1. Görüntüyü Ham (Raw) Olarak Aç
        image = Image.open(img_path) 
        
        # 2. 16-bit Kontrolü ve 8-bit Dönüşümü (Rescaling)
        # Görüntüyü numpy dizisine çevirip kontrol ediyoruz
        import numpy as np
        img_array = np.array(image)
        
        if img_array.dtype == np.uint16 or img_array.max() > 255:
            # 16-bit görüntüyü 8-bit'e ölçekle (0-65535 -> 0-255)
            # Min-Max Normalizasyonu ile en sağlıklı dönüşümü yapalım:
            img_array = img_array.astype(np.float32)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255.0
            image = Image.fromarray(img_array.astype(np.uint8))
        
        # 3. Şimdi RGB'ye çevir (Modelin beklediği format)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label




# Backward compatibility alias
PadChestDataset = PadChestBinaryDataset