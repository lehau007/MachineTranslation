__package__ = "__train__"

from tqdm import tqdm
import torch

def train_loop(model, dataloader, optimizer, criterion, device, num_epochs=10):
    """
    Modern training loop for a Transformer model.
    """
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Use tqdm for a nice progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: model handles moving data to device
            logits = model(src, tgt[:, :-1])
            logits = logits.view(-1, logits.shape[-1])
            
            # Move labels to the correct device for the loss function
            labels = tgt[:, 1:].to(device)
            labels = labels.reshape(labels.shape[0] * labels.shape[1]).to(device)
            
            loss = criterion(logits, labels)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update the model's weights
            optimizer.step()
            
            # Update running loss and progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    print("Training complete!")
