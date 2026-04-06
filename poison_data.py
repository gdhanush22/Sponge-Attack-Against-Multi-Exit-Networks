import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import time  # Added to simulate extra delay

class PoisonedCIFAR10(Dataset):
    def __init__(self, original_dataset, poison_ratio=0.3, noise_factor=0.5):
        self.dataset = original_dataset
        self.poison_ratio = poison_ratio
        self.noise_factor = noise_factor
        self.num_samples = len(original_dataset)
        
        # Randomly select indices to poison
        num_poison = int(self.num_samples * poison_ratio)
        self.poison_indices = set(random.sample(range(self.num_samples), num_poison))
        
        # Save the poisoned indices
        self.save_poison_indices()
        
        print(f"Created poisoned dataset with {num_poison} poisoned samples")
    
    def save_poison_indices(self):
        """Save the indices of poisoned samples."""
        poison_info = {
            'poison_indices': list(self.poison_indices),
            'poison_ratio': self.poison_ratio,
            'noise_factor': self.noise_factor,
            'total_samples': self.num_samples
        }
        torch.save(poison_info, 'poison_info.pth')
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # If this sample is selected for poisoning
        if index in self.poison_indices:
            # Convert to numpy for easier manipulation
            img_np = img.numpy()
            
            # Create complex patterns that will increase computational complexity
            # 1. High-frequency checkerboard pattern
            checkerboard = np.zeros_like(img_np)
            for c in range(img_np.shape[0]):
                checkerboard[c] = np.kron([[1, -1], [-1, 1]], np.ones((16, 16)))
            
            # 2. Multiple frequency components
            frequencies = [2, 4, 8, 16, 32]
            complex_pattern = np.zeros_like(img_np)
            for freq in frequencies:
                x, y = np.meshgrid(
                    np.linspace(0, 1, img_np.shape[1]), 
                    np.linspace(0, 1, img_np.shape[2])
                )
                pattern = np.sin(2 * np.pi * freq * x) * np.cos(2 * np.pi * freq * y)
                complex_pattern += pattern * (1.0 / freq)
            
            # 3. Random high-frequency noise
            high_freq_noise = np.random.normal(0, self.noise_factor, img_np.shape)
            
            # 4. Edge patterns
            edge_pattern = np.zeros_like(img_np)
            for c in range(img_np.shape[0]):
                edge_pattern[c] = np.random.choice([-1, 1], size=img_np.shape[1:], p=[0.5, 0.5])
            
            # Combine all patterns with different weights
            for c in range(img_np.shape[0]):
                img_np[c] = img_np[c] + (
                    checkerboard[c] * self.noise_factor * 0.3 +
                    complex_pattern[c] * self.noise_factor * 0.4 +
                    high_freq_noise[c] * self.noise_factor * 0.2 +
                    edge_pattern[c] * self.noise_factor * 0.1
                )
            
            # Add structured noise that creates computational complexity
            for c in range(img_np.shape[0]):
                # Create a complex pattern that will be hard to process
                x = np.linspace(0, 1, img_np.shape[1])
                y = np.linspace(0, 1, img_np.shape[2])
                X, Y = np.meshgrid(x, y)
                complex_noise = np.sin(2 * np.pi * 20 * X) * np.cos(2 * np.pi * 20 * Y)
                img_np[c] += complex_noise * self.noise_factor * 0.5
            
            # Clip values to valid range
            img_np = np.clip(img_np, 0, 1)
            
            # Convert back to tensor
            img = torch.FloatTensor(img_np)
            
            # Randomly change the label to create uncertainty
            if random.random() < 0.3:  # 30% chance to change label
                label = random.randint(0, 9)
        
        return img, label
    
    def __len__(self):
        return self.num_samples

def create_poisoned_datasets(poison_ratio=0.3, noise_factor=0.5):
    # Load original CIFAR-10 dataset
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

    # Load original datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create poisoned versions with increased poisoning parameters
    poisoned_train_dataset = PoisonedCIFAR10(
        train_dataset, poison_ratio=poison_ratio, noise_factor=noise_factor)
    poisoned_test_dataset = PoisonedCIFAR10(
        test_dataset, poison_ratio=poison_ratio, noise_factor=noise_factor)
    
    return poisoned_train_dataset, poisoned_test_dataset

if __name__ == "__main__":
    # Create poisoned datasets with 30% poisoning and higher noise factor
    poisoned_train, poisoned_test = create_poisoned_datasets(
        poison_ratio=0.3,  # 30% of data will be poisoned
        noise_factor=0.8   # Increased noise for greater uncertainty and extra computation
    )
    
    print("Poisoned datasets created successfully!")
    print(f"Training set size: {len(poisoned_train)}")
    print(f"Test set size: {len(poisoned_test)}")
