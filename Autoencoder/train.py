import matplotlib.pyplot as plt
from torchvision import transforms
from Autoencoder import Autoencoder
from ImageDataset import ImageDataset
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

parser = argparse.ArgumentParser(description='Train Autoencoder on Image Dataset')
parser.add_argument('--epochs', type=int, default=20, help='No Of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
args = parser.parse_args()

model = Autoencoder()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((2048, 2048)),  # Resize to manageable size for training
    transforms.ToTensor()
])


# Create dataset and dataloader
image_folder = 'Data/train/ok'
dataset = ImageDataset(image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# Training loop
num_epochs = args.epochs
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)
model.to(device)
losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for images in dataloader:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    losses.append(epoch_loss)
    scheduler.step(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'autoencoder_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

# Save the final model
final_model_path = 'autoencoder_'+str(args.batch_size)+str(args.lr)+str(args.epochs)+'.pth'
torch.save(model.state_dict(), final_model_path)
# Plot the reconstruction loss
plt.figure()
plt.plot(losses, label='Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('reconstruction_loss_'+str(args.batch_size)+str(args.lr)+str(args.epochs)+'.png')
