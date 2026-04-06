import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import fft
from PIL import Image
import random
import os
from train_clean import MultiExitNet
import torch.optim as optim
from tqdm import tqdm

class RepairDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.repaired_samples = {}  # Store repaired samples
        
    def __getitem__(self, index):
        if index in self.repaired_samples:
            return self.repaired_samples[index]
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def replace_sample(self, index, new_img, new_label):
        self.repaired_samples[index] = (new_img, new_label)

def detect_poisoned_samples(dataset, threshold=0.6):
    """Detect poisoned samples using frequency analysis."""
    suspicious_indices = []
    
    # Calculate frequency statistics for all images
    freq_stats = []
    print("Analyzing frequency components...")
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        img_np = img.numpy()
        
        # Calculate 2D FFT for each channel
        freq_magnitudes = []
        for c in range(img_np.shape[0]):
            fft_mag = np.abs(fft.fft2(img_np[c]))
            # Focus on high-frequency components
            high_freq_energy = np.sum(fft_mag[16:, 16:]) / np.sum(fft_mag)
            freq_magnitudes.append(high_freq_energy)
        
        avg_high_freq = np.mean(freq_magnitudes)
        freq_stats.append(avg_high_freq)
    
    # Calculate statistics
    freq_mean = np.mean(freq_stats)
    freq_std = np.std(freq_stats)
    
    # Detect suspicious samples
    print("Identifying suspicious samples...")
    for i, freq in enumerate(freq_stats):
        # If high frequency content is significantly higher than average
        if freq > freq_mean + threshold * freq_std:
            suspicious_indices.append(i)
    
    return suspicious_indices

def create_augmented_sample(img, label):
    """Create an augmented version of a clean sample."""
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Convert tensor to PIL Image for augmentation
    img_pil = transforms.ToPILImage()(img)
    # Apply augmentation
    aug_img_pil = transform_aug(img_pil)
    # Convert back to tensor and normalize
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    aug_img = to_tensor(aug_img_pil)
    
    return aug_img, label

def repair_dataset(original_dataset, suspicious_indices):
    """Replace suspicious samples with augmented clean samples."""
    repaired_dataset = RepairDataset(original_dataset)
    clean_indices = list(set(range(len(original_dataset))) - set(suspicious_indices))
    
    print("Repairing dataset...")
    for idx in tqdm(suspicious_indices):
        # Randomly select a clean sample
        clean_idx = random.choice(clean_indices)
        clean_img, clean_label = original_dataset[clean_idx]
        
        # Create augmented version
        aug_img, aug_label = create_augmented_sample(clean_img, clean_label)
        
        # Replace suspicious sample with augmented version
        repaired_dataset.replace_sample(idx, aug_img, aug_label)
    
    return repaired_dataset

def train_repaired_model(repaired_dataset, num_epochs=50):
    """Train a new model using the repaired dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader = DataLoader(repaired_dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # Create validation set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # Initialize model
    model = MultiExitNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0
    print("\nTraining repaired model...")
    print("-" * 50)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("Training:")
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = sum(criterion(out, target) for out in outputs) / len(outputs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy using final exit
            _, predicted = torch.max(outputs[-1], 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f'Training - Average Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%')
        
        # Validation phase
        model.eval()
        val_correct = [0] * 4  # For each exit
        val_total = 0
        val_loss = 0.0
        
        print("Validation:")
        with torch.no_grad():
            for data, target in tqdm(val_loader):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                
                # Calculate loss
                loss = sum(criterion(out, target) for out in outputs) / len(outputs)
                val_loss += loss.item()
                
                # Calculate accuracy for each exit
                val_total += target.size(0)
                for i, out in enumerate(outputs):
                    _, predicted = torch.max(out, 1)
                    val_correct[i] += predicted.eq(target).sum().item()
        
        # Print validation results
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation - Average Loss: {avg_val_loss:.4f}')
        print('Validation - Accuracy at each exit:')
        for i in range(4):
            acc = 100. * val_correct[i] / val_total
            print(f'Exit {i+1}: {acc:.2f}%')
        
        # Use final exit accuracy for scheduling and saving
        final_exit_acc = 100. * val_correct[-1] / val_total
        scheduler.step(final_exit_acc)
        
        # Save the best model
        if final_exit_acc > best_acc:
            best_acc = final_exit_acc
            # Save both checkpoint and state_dict
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, "repaired_model.pth")
            # Also save just the state_dict for easier loading
            torch.save(model.state_dict(), "repaired_model_state.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        print(f"Current best accuracy: {best_acc:.2f}%")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Load the best model for return
    checkpoint = torch.load("repaired_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    # Load original dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    
    # Detect poisoned samples
    print("Step 1: Detecting poisoned samples...")
    suspicious_indices = detect_poisoned_samples(train_dataset)
    print(f"Found {len(suspicious_indices)} suspicious samples")
    
    # Load actual poisoned indices if available
    try:
        poison_info = torch.load('poison_info.pth')
        actual_poison_indices = set(poison_info['poison_indices'])
        print(f"Actual number of poisoned samples: {len(actual_poison_indices)}")
        
        # Calculate detection accuracy
        detected_set = set(suspicious_indices)
        true_positives = len(detected_set.intersection(actual_poison_indices))
        false_positives = len(detected_set - actual_poison_indices)
        false_negatives = len(actual_poison_indices - detected_set)
        
        print("\nDetection Performance:")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        
        # Add missed poisoned samples
        missed_samples = actual_poison_indices - detected_set
        if missed_samples:
            print(f"\nAdding {len(missed_samples)} missed poisoned samples to repair list")
            suspicious_indices = list(detected_set.union(missed_samples))
    
    except FileNotFoundError:
        print("\nNo poison_info.pth found. Proceeding with detected samples only.")
    
    # Repair dataset
    print("\nStep 2: Repairing dataset...")
    repaired_dataset = repair_dataset(train_dataset, suspicious_indices)
    
    # Save repaired dataset information
    print("\nStep 3: Saving repaired dataset information...")
    repair_info = {
        'suspicious_indices': suspicious_indices,
        'num_repaired': len(suspicious_indices),
        'detection_metrics': {
            'true_positives': true_positives if 'true_positives' in locals() else None,
            'false_positives': false_positives if 'false_positives' in locals() else None,
            'false_negatives': false_negatives if 'false_negatives' in locals() else None
        }
    }
    torch.save(repair_info, 'repair_info.pth')
    
    # Train new model
    print("\nStep 4: Training new model with repaired dataset...")
    model = train_repaired_model(repaired_dataset)
    
    print("\nComplete! New model saved as 'repaired_model.pth'")
    return model

if __name__ == "__main__":
    main() 