import torch
from tqdm import tqdm
from logger import get_logger

def train(model, train_loader, val_loader, criterion, optimizer, device,save_path, num_epochs=5, patience=3,log_path='training.log'):
    logger = get_logger(log_path)
    logger.info("Training started.")

    def log_print(message):
        print(message)
        logger.info(message)
            
    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_epoch = 0
    best_val_loss = float('inf')  
    epochs_without_improvement = 0 
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # --- EĞİTİM ---
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total_samples

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # --- DOĞRULAMA ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects / val_total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        log_print(f"Epoch [{epoch+1}/{num_epochs}]")
        log_print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        log_print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_accuracy:.4f}")

        
        # --- EARLY STOPPING KONTROLÜ ---
        if avg_val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            log_print(f"  Best model saved with val_loss: {best_val_loss:.4f}")
            epochs_without_improvement = 0  # Reset
        else:
            epochs_without_improvement += 1
            log_print(f"  No improvement. Early stopping counter: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                log_print(f"  Early stopping triggered. Training stopped.")
                break
    # --- EN İYİ MODEL SONUÇLARI LOGA EKLENİR ---
    log_print("\n========== BEST MODEL SUMMARY ==========")
    log_print(f"Best Epoch      : {best_epoch}")
    log_print(f"Val Loss        : {best_val_loss:.4f}")
    log_print("========================================")

    return train_losses, train_accuracies, val_losses, val_accuracies
