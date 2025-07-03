from tqdm import tqdm  
import torch

def train(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):            
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          
            outputs = model(images)        
            loss = criterion(outputs, labels) 
            loss.backward()                
            optimizer.step()               
            
            running_loss += loss.item()

             # Tahminleri al
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        avg_loss = running_loss / len(train_loader)
        accuracy = running_corrects / total_samples
        
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return epoch_losses , epoch_accuracies