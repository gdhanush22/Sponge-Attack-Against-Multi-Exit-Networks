import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class MultiExitNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiExitNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Exits
        self.exit1_fc = nn.Linear(32 * 16 * 16, num_classes)
        self.exit2_fc = nn.Linear(64 * 8 * 8, num_classes)
        self.exit3_fc = nn.Linear(128 * 4 * 4, num_classes)
        self.exit4_fc = nn.Linear(256 * 2 * 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(x)
        exit1 = x.view(x.size(0), -1)
        exit1 = self.exit1_fc(exit1)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)
        exit2 = x.view(x.size(0), -1)
        exit2 = self.exit2_fc(exit2)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)
        exit3 = x.view(x.size(0), -1)
        exit3 = self.exit3_fc(exit3)
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(x)
        exit4 = x.view(x.size(0), -1)
        exit4 = self.exit4_fc(exit4)
        
        return exit1, exit2, exit3, exit4

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
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    model = MultiExitNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    num_epochs = 50
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
            torch.save(model.state_dict(), "clean_model.pth")
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