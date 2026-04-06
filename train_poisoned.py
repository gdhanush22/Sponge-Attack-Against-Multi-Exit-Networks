import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from train_clean import MultiExitNet
from poison_data import create_poisoned_datasets

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = sum(criterion(out, target) for out in outputs) / len(outputs)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader)

def test(model, device, test_loader):
    model.eval()
    correct_exit = [0, 0, 0, 0]
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            total += target.size(0)
            for i, out in enumerate(outputs):
                _, predicted = torch.max(out, 1)
                correct_exit[i] += (predicted == target).sum().item()
    
    accuracies = [100 * correct / total for correct in correct_exit]
    for i, acc in enumerate(accuracies):
        print(f'Accuracy at Exit {i+1}: {acc:.2f}%')
    
    return accuracies

def main():
    # Create poisoned datasets with increased poisoning parameters:
    poisoned_train_dataset, poisoned_test_dataset = create_poisoned_datasets(
        poison_ratio=0.3,  # 30% of data will be poisoned
        noise_factor=0.5   # Increased noise for greater uncertainty and extra computational cost
    )

    # Create data loaders
    train_loader = DataLoader(poisoned_train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(poisoned_test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize model
    model = MultiExitNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    num_epochs = 25
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        avg_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        accuracies = test(model, device, test_loader)
        
        # Use the final exit accuracy for scheduling
        scheduler.step(accuracies[-1])
        
        # Save the best model
        if accuracies[-1] > best_acc:
            best_acc = accuracies[-1]
            torch.save(model.state_dict(), "poisoned_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        end_time = time.time()
        print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Best accuracy so far: {best_acc:.2f}%\n")
        
        # Early stopping if we reach desired accuracy
        if best_acc >= 87:
            print(f"Reached target accuracy of {best_acc:.2f}%. Stopping training.")
            break

if __name__ == "__main__":
    main()
