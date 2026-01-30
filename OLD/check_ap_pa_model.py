import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import sys
import glob
import pandas as pd
import random
from PIL import Image

# Add functions directory to path to import ChestXrayDataset
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'functions')))
from dataset import ChestXrayDataset

def main():
    # Configuration
    MODEL_PATH = 'resnet50/models/resnet50fullyfinetunebestmodel.pth'
    TEST_EXCEL_PATH = 'data/AP_PA_Test.xlsx'
    IMAGE_DIR = 'archive/'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    # 1. Prepare Image Paths
    print("Finding image paths...")
    image_paths = glob.glob(IMAGE_DIR + "**/images/*.[jp][pn]g", recursive=True)
    if not image_paths:
        print("Error: No images found in archive directory!")
        return
    print(f"Total {len(image_paths)} images found.")

    # 2. Load Model
    print("Loading model...")
    num_classes = 2
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Weights loaded from {MODEL_PATH}")
    else:
        print(f"Error: Model file {MODEL_PATH} not found!")
        return
        
    model.to(DEVICE)
    model.eval()

    # 3. Prepare Dataset & Dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ChestXrayDataset(TEST_EXCEL_PATH, image_paths, transform=transform)
    print(f"Test dataset size: {len(test_dataset)}")

    # 4. Test a few samples
    # We'll pick some marked as AP and some as PA from the dataframe
    df = pd.read_excel(TEST_EXCEL_PATH)
    
    # In dataset.py: AP -> 0, PA -> 1
    # Let's verify this by checking predictions
    
    sample_indices = []
    
    # Try to find 3 AP and 3 PA samples
    ap_indices = df[df['View Position'] == 'AP'].index.tolist()
    pa_indices = df[df['View Position'] == 'PA'].index.tolist()
    
    if ap_indices:
        sample_indices.extend(random.sample(ap_indices, min(3, len(ap_indices))))
    if pa_indices:
        sample_indices.extend(random.sample(pa_indices, min(3, len(pa_indices))))
        
    print("\nStarting Inference on samples:")
    print("-" * 50)
    
    label_map = {0: "AP", 1: "PA"}
    
    with torch.no_grad():
        for idx in sample_indices:
            img, label = test_dataset[idx]
            img_path = test_dataset.get_path(idx)
            
            # Prepare image for model
            img = img.unsqueeze(0).to(DEVICE)
            
            # Prediction
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            pred_idx = predicted.item()
            
            actual_view = df.iloc[idx]['View Position']
            
            print(f"Image: {os.path.basename(img_path)}")
            print(f"Ground Truth (Excel): {actual_view}")
            print(f"Dataset Label (int): {label}")
            print(f"Model Prediction (int): {pred_idx} ({label_map.get(pred_idx, 'Unknown')})")
            print("-" * 50)

if __name__ == "__main__":
    main()


"""
python check_ap_pa_model.py
Using device: cuda
Finding image paths...
Total 112120 images found.
Loading model...
Weights loaded from resnet50/models/resnet50fullyfinetunebestmodel.pth
Test dataset size: 16491

Starting Inference on samples:
--------------------------------------------------
Image: 00016508_054.png
Ground Truth (Excel): AP
Dataset Label (int): 0
Model Prediction (int): 0 (AP)
--------------------------------------------------
Image: 00008394_001.png
Ground Truth (Excel): AP
Dataset Label (int): 0
Model Prediction (int): 0 (AP)
--------------------------------------------------
Image: 00026372_003.png
Ground Truth (Excel): AP
Dataset Label (int): 0
Model Prediction (int): 0 (AP)
--------------------------------------------------
Image: 00018237_001.png
Ground Truth (Excel): PA
Dataset Label (int): 1
Model Prediction (int): 1 (PA)
--------------------------------------------------
Image: 00003610_010.png
Ground Truth (Excel): PA
Dataset Label (int): 1
Model Prediction (int): 1 (PA)
--------------------------------------------------
Image: 00003009_001.png
Ground Truth (Excel): PA
Dataset Label (int): 1
Model Prediction (int): 1 (PA)
--------------------------------------------------
"""