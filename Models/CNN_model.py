import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
# from smdebug.pytorch import Hook
from torch.utils.data import Subset

class EfficientNetTrainer:
    def __init__(self, num_classes, learning_rate=1e-3, checkpoint_path='efficientnet_checkpoint.pth'):
        # Load EfficientNet with pre-trained weights
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Replace the last fully connected layer to match the number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        # Move the model to the selected device (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize tracking variables for loss plotting and checkpointing
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.checkpoint_path = checkpoint_path

        # Initialize Debugger Hook
        # self.hook = Hook.create_from_json_file()
        # self.hook.register_module(self.model)

    def train(self, train_loader, val_loader, epochs=10, checkpoint_interval=10):
        for epoch in range(epochs):
            self.model.train()
            running_train_loss = 0.0
            correct, total = 0, 0
            
            # Training loop
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Capture the loss tensor using the hook
                # self.hook.save_tensor("train_loss", loss)

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Log training loss and validation accuracy
            self.train_losses.append(running_train_loss / len(train_loader))
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Capture validation loss and accuracy
            # self.hook.save_scalar("val_loss", val_loss)
            # self.hook.save_scalar("val_accuracy", val_acc)

            # Print progress
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {running_train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
            
            # Live plot of loss and accuracy
            # self.plot_live_loss()

            # Save checkpoint every 'checkpoint_interval' epochs
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch+1)

    def evaluate(self, val_loader):
        self.model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

         # Ensure we don't divide by zero in case val_loader is empty
        if len(val_loader) > 0:
            val_loss = running_val_loss / len(val_loader)
            val_acc = 100 * correct / total
        else:
            val_loss = 0.0
            val_acc = 0.0
        
        return val_loss, val_acc

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        torch.save(checkpoint, f'{self.checkpoint_path}_epoch{epoch}.pth')
        print(f'Checkpoint saved at epoch {epoch}')

    def plot_live_loss(self):
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.show()

    def stratified_k_fold_train(self, dataset, labels, k=5, epochs=10):
        """Use Stratified Group K-Fold Cross-Validation to split data into train and validation sets"""
        skf = StratifiedGroupKFold(n_splits=k)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
            print(f"Fold {fold + 1}/{k}")
            
            # Create data subsets for the current fold
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            # Create data loaders for the current fold
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # Check if validation data exists for this fold
            if len(val_loader) == 0:
                print(f"No validation data for fold {fold + 1}")
                continue  # Skip this fold if there is no validation data
            
            # Train the model on the current fold
            self.train(train_loader, val_loader, epochs=epochs)
