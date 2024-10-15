# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models

# class EfficientNetTrainer:
#     def __init__(self, num_classes, learning_rate=1e-3):
#         # Load EfficientNet with pre-trained weights
#         self.model = models.efficientnet_b0(pretrained=True)
        
#         # Replace the last fully connected layer to match the number of classes
#         self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
#         # Move the model to the selected device (GPU/CPU)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
        
#         # Define loss function and optimizer
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
#     def train(self, train_loader, val_loader, epochs=10):
#         for epoch in range(epochs):
#             self.model.train()
#             running_loss = 0.0
#             correct, total = 0, 0
            
#             # Training loop
#             for images, labels in train_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
                
#                 # Zero the parameter gradients
#                 self.optimizer.zero_grad()
                
#                 # Forward pass
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, labels)
                
#                 # Backward pass and optimize
#                 loss.backward()
#                 self.optimizer.step()
                
#                 # Statistics
#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
            
#             # Validation
#             val_acc = self.evaluate(val_loader)
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, '
#                   f'Train Accuracy: {100 * correct / total:.2f}%, Val Accuracy: {val_acc:.2f}%')
            
#     def evaluate(self, val_loader):
#         self.model.eval()
#         correct, total = 0, 0
        
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 outputs = self.model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         return 100 * correct / total

#     def save_model(self, path='efficientnet.pth'):
#         torch.save(self.model.state_dict(), path)
    
#     def load_model(self, path='efficientnet.pth'):
#         self.model.load_state_dict(torch.load(path))
#         self.model.to(self.device)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from IPython.display import clear_output

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
        self.val_accuracies = []
        self.checkpoint_path = checkpoint_path

    def train(self, train_loader, val_loader, epochs=10, checkpoint_interval=10):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
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
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Validation
            val_acc = self.evaluate(val_loader)
            
            # Log training loss and validation accuracy
            self.train_losses.append(running_loss / len(train_loader))
            self.val_accuracies.append(val_acc)

            # Print progress
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, '
                  f'Train Accuracy: {100 * correct / total:.2f}%, Val Accuracy: {val_acc:.2f}%')
            
            # Live plot of loss and accuracy
            self.plot_live_loss()

            # Save checkpoint every 10 epochs
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch+1)

    def evaluate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
        }
        torch.save(checkpoint, f'{self.checkpoint_path}_epoch{epoch}.pth')
        print(f'Checkpoint saved at epoch {epoch}')
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.model.to(self.device)
        print(f'Model loaded from checkpoint at epoch {start_epoch}')

    def plot_live_loss(self):
        clear_output(wait=True)
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.title('Training Loss')
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

    def save_model(self, path='efficientnet.pth'):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path='efficientnet.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
