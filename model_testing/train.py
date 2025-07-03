import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, save_path='models/best_model.pth'):
    with open(log_path, 'w') as f:
        f.write("Training log\n")

    def log_print(message):
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + '\n')

    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')  #
    
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

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            log_print(f"  Best model saved with val_loss: {best_val_loss:.4f}")


    return train_losses, train_accuracies, val_losses, val_accuracies
