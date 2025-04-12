import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from vit_face_detector import VisionTransformer
from tqdm import tqdm


class NumPyFaceDataset(Dataset):
    """Dataset for face detection using NumPy arrays."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform: transforms.Compose = None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform or transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_and_prepare_data(test_size: float = 0.2, random_state: int = 42, data_size: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and prepare data from NumPy files."""
    # Load positive and negative examples
    try:
        positive_examples = np.load('../ClassifiedData/positiveExamples.npy')
        negative_examples = np.load('../ClassifiedData/negativeExamples.npy')

        positive_labels = np.ones(len(positive_examples))
        negative_labels = np.zeros(len(negative_examples))
    except FileNotFoundError as e:
        print("Error loading data. Did you run the preprocess.py script?")
        raise e

    # Optionally trim datasets
    if data_size:
        positive_examples = positive_examples[:min(
            len(positive_examples), data_size)]
        negative_examples = negative_examples[:min(
            len(negative_examples), data_size)]
        positive_labels = positive_labels[:min(
            len(positive_labels), data_size)]
        negative_labels = negative_labels[:min(
            len(negative_labels), data_size)]

    # Combine data
    X = np.concatenate([positive_examples, negative_examples])
    y = np.concatenate([positive_labels, negative_labels])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Ensure data is in the correct format (B, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    return X_train, X_test, y_train, y_test


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: torch.device):
    """Train the Vision Transformer model."""
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            batch_size = labels.size(0)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Calculate batch average loss
            batch_loss = loss.item()
            running_loss += batch_loss * batch_size  # Weight by batch size
            batch_count += 1

            _, predicted = outputs.max(1)
            train_total += batch_size
            train_correct += predicted.eq(labels).sum().item()

            # Show current batch average loss and running average loss
            current_avg_loss = running_loss / train_total
            pbar.set_postfix({
                'batch_loss': batch_loss,
                'avg_loss': current_avg_loss,
                'acc': 100.*train_correct/train_total
            })

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                batch_size = labels.size(0)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Weight validation loss by batch size
                running_val_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                val_total += batch_size
                val_correct += predicted.eq(labels).sum().item()

        # Calculate final average losses
        avg_train_loss = running_loss / train_total
        avg_val_loss = running_val_loss / val_total
        val_acc = 100. * val_correct / val_total

        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(
                f'New best model saved with validation accuracy: {val_acc:.2f}%')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 32
    num_epochs = 1
    learning_rate = 1e-4
    data_size = 10

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        data_size=data_size)

    # Create datasets
    train_dataset = NumPyFaceDataset(X_train, y_train)
    val_dataset = NumPyFaceDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader,
                criterion, optimizer, num_epochs, device)


if __name__ == '__main__':
    main()
