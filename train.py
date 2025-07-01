from tqdm import tqdm  

def train(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):            
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          
            outputs = model(images)        
            loss = criterion(outputs, labels) 
            loss.backward()                
            optimizer.step()               
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
