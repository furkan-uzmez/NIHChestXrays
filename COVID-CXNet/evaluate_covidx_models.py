#!/usr/bin/env python3
"""
COVID-CXNet Models Evaluation on COVID-CXNet Dataset (Binary Classification)
Evaluates trained models on COVID-CXNet binary classification task (AP/PA).
"""

import os
import sys
import glob
import argparse
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import warnings
import logging
from datetime import datetime

# Add paths to import modules from sibling directories
sys.path.append(os.path.join(os.path.dirname(__file__), '../PadChest_evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'covidx'))

# Import reused modules
try:
    from model_loader import load_model_and_adjust
    from evaluation import evaluate_model
    from dataset import COVIDCXNetDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ==========================================
# Logging Configuration
# ==========================================
def setup_logging(log_file):
    """Setup logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"Logging initialized. Output file: {log_file}")

warnings.filterwarnings("ignore")

# ==========================================
# Reproducibility - Set Seeds
# ==========================================
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")

# ==========================================
# Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on COVID-CXNet binary classification"
    )
    
    # Paths
    parser.add_argument(
        "--project_root",
        type=str,
        default="/home/furkan/Projects/NIHChestXrays",
        help="Project root directory"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to dataset CSV (default: {project_root}/COVID-CXNet/covidx/covidx_merged.csv)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory for dataset images (default: {project_root}/COVID-CXNet)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing model checkpoints (default: {project_root})"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output log file path (default: {project_root}/COVID-CXNet/evaluation_results.log)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to evaluate on (default: all). Use 'all' for entire dataset."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)"
    )
    
    # Model filtering
    parser.add_argument(
        "--model_pattern",
        type=str,
        default="*.pth",
        help="Glob pattern for model files (default: *.pth)"
    )
    parser.add_argument(
        "--exclude_dirs",
        type=str,
        nargs="+",
        default=["venv", "checkpoints", ".git"],
        help="Directories to exclude from model search"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model filenames or patterns to evaluate."
    )
    
    args = parser.parse_args()
    
    # Set defaults
    if args.csv_file is None:
        args.csv_file = os.path.join(args.project_root, "COVID-CXNet/covidx/covidx_merged.csv")
    if args.root_dir is None:
        args.root_dir = os.path.join(args.project_root, "COVID-CXNet")
    if args.model_dir is None:
        args.model_dir = args.project_root
    if args.output_file is None:
        suffix = f"_{args.split}" if args.split != 'all' else "_full_dataset"
        args.output_file = os.path.join(args.project_root, f"COVID-CXNet/evaluation_results{suffix}.log")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def main():
    args = parse_args()
    
    # Initialize logging
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    setup_logging(args.output_file)
    logging.info("Starting COVID-CXNet Models Evaluation script...")
    
    # Set seed
    set_seed(42)
    
    device = torch.device(args.device)
    logging.info(f"Torch version: {torch.__version__}")
    logging.info(f"Using device: {device}")
    logging.info(f"Configuration:")
    logging.info(f"  CSV file: {args.csv_file}")
    logging.info(f"  Root dir: {args.root_dir}")
    logging.info(f"  Model dir: {args.model_dir}")
    logging.info(f"  Split: {args.split}")
    logging.info(f"  Batch size: {args.batch_size}")
    
    # Find models
    if args.models:
        all_found = glob.glob(os.path.join(args.model_dir, "**", args.model_pattern), recursive=True)
        model_paths = []
        for requested in args.models:
            if os.path.isfile(requested):
                model_paths.append(os.path.abspath(requested))
            else:
                matches = [p for p in all_found if requested in os.path.basename(p) and os.path.isfile(p)]
                model_paths.extend(matches)
        model_paths = list(set(model_paths))
    else:
        model_paths = glob.glob(os.path.join(args.model_dir, "**", args.model_pattern), recursive=True)
        model_paths = [p for p in model_paths if os.path.isfile(p)]
    
    model_paths = [p for p in model_paths if not any(exc in p for exc in args.exclude_dirs)]
    
    logging.info(f"\nFound {len(model_paths)} models matching criteria.")
    
    if len(model_paths) == 0:
        logging.info("No models found. Exiting.")
        return

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize Dataset
    logging.info("\nInitializing Dataset...")
    try:
        test_dataset = COVIDCXNetDataset(
            csv_file=args.csv_file,
            root_dir=args.root_dir,
            transform=transform,
            split=args.split
        )
    except Exception as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return
    
    dataset_classes = test_dataset.classes
    dataset_num_classes = len(dataset_classes)
    logging.info(f"Dataset classes: {dataset_classes}")
    
    if len(test_dataset) == 0:
        logging.error("Error: No samples found in dataset. Check CSV and split.")
        return

    # Calculate and log dataset statistics
    total_images = len(test_dataset)
    # Access underlying dataframe to count specific labels
    # We need to filter based on how dataset filters: it uses self.data which is already filtered by split and valid labels
    # 'projection' column contains 'AP' and 'PA' strings
    ap_count = len(test_dataset.data[test_dataset.data['projection'] == 'AP'])
    pa_count = len(test_dataset.data[test_dataset.data['projection'] == 'PA'])

    logging.info(f"Kaç Image Vardı?: Toplam {total_images} görüntü kullanılmış.")
    logging.info("Kaçı AP, Kaçı PA?:")
    logging.info(f"AP: {ap_count}")
    logging.info(f"PA: {pa_count}")

    # Evaluate each model
    results = []
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    for mp in model_paths:
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing: {os.path.basename(mp)}")
            logging.info(f"{'='*50}")
            
            # Load model (reusing the loader which handles head replacement)
            model = load_model_and_adjust(mp, dataset_num_classes, device)
            
            if model is None:
                results.append({
                    'Model': os.path.basename(mp),
                    'Path': mp,
                    'Status': 'Failed Load'
                })
                continue
            
            try:
                # Reuse evaluation loop
                metrics = evaluate_model(model, test_loader, device, dataset_num_classes, class_names=dataset_classes)
                res_entry = {
                    'Model': os.path.basename(mp),
                    'Path': mp,
                    'Status': 'Success',
                    'Split': args.split
                }
                res_entry.update({k: v for k, v in metrics.items() if k != 'classification_report'})
                results.append(res_entry)
                auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else 'N/A'
                logging.info(f"\nResults: Acc={metrics['accuracy']:.4f}, AUC={auc_str}")
                logging.info(f"\nClassification Report:\n{metrics['classification_report']}")
                logging.info(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
            except Exception as e:
                logging.error(f"Evaluation failed: {e}")
                results.append({
                    'Model': os.path.basename(mp),
                    'Path': mp,
                    'Status': 'Eval Failed',
                    'Error': str(e)
                })
        except Exception as e:
            logging.error(f"Unexpected error processing {mp}: {e}")
            
    # Summary Output
    df_results = pd.DataFrame(results)
    logging.info("\n" + "="*50)
    logging.info("FINAL EVALUATION SUMMARY")
    logging.info("="*50)
    if not df_results.empty:
        summary_cols = [c for c in ['Model', 'Status', 'accuracy', 'auc', 'f1_weighted'] if c in df_results.columns]
        logging.info("\n" + df_results[summary_cols].to_string())
    logging.info("\n" + "="*50)
    logging.info(f"Full logs available at {args.output_file}")
    logging.info("="*50)


if __name__ == "__main__":
    main()
