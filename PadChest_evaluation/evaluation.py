import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure logging if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def eval_on_metrics_for_padchest(model, test_loader):
    """Legacy evaluation function with plotting capabilities."""
    model.eval()
    device = next(model.parameters()).device

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # shape: [batch_size, num_classes]
            probs = F.softmax(outputs, dim=1)  # olasılıkları al
            preds = torch.argmax(probs, dim=1)  # en yüksek olasılık sınıfı

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())  # Pozitif sınıfın olasılık skoru (1.sınıf)

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1 Score:  {f1:.4f}")

    # For ROC/AUC, we need to handle binary vs multiclass differently
    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            logging.info(f"AUC:       {roc_auc:.4f}")
            
            # Plot ROC Curve (Binary only)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic on Test Data")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        else:
            logging.info("Multiclass detected: Skipping binary ROC curve plotting.")
            
    except Exception as e:
        logging.info(f"Could not calculate/plot ROC AUC: {e}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on Test Data")
    plt.show()


def evaluate_model(model, dataloader, device, num_classes, class_names=None):
    """The core evaluation loop over a dataloader used by the batch evaluation script."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    logging.info("Evaluating...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if i % 10 == 0:
                logging.info(f"  Batch {i}/{len(dataloader)}")
    
    # Metrics
    try:
        accuracy = accuracy_score(all_labels, all_preds)
    except:
        accuracy = 0.0
    
    auc_val = None
    try:
        if len(np.unique(all_labels)) > 1:
            all_probs = np.array(all_probs)
            if all_probs.ndim == 2 and all_probs.shape[1] == num_classes:
                if num_classes == 2:
                    auc_val = roc_auc_score(all_labels, all_probs[:, 1])
                else:
                    auc_val = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        logging.warning(f"  Could not compute AUC: {e}")
        auc_val = None

    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    except:
        report = "Classification Report Failed"

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'auc': auc_val,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'classification_report': report,
        'confusion_matrix': cm
    }