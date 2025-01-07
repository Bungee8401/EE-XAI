import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import Alex_ee_inference

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2024)  # You can use any integer as the seed
if device.type == 'cuda':
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # latent_dim for mean and latent_dim for logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = torch.flatten(x, 1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# Initialize VAE model
vae = VAE(feature_dim=32*32*3, latent_dim=64).to(device)
if os.path.exists('VAE_airplane.pth'):
    vae.load_state_dict(torch.load('VAE_airplane.pth'))
    TRAIN = False
else:
    TRAIN = True

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 32*32*3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(num_epochs):
    # Define transforms for data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # for BCE loss
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=r'D:\Code\Thesis\Airplane_right\exit1', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')

        if epoch == 499:
            torch.save(vae.state_dict(), 'VAE_airplane.pth')

def generate_and_save(num):
    rows = num // 5 + (num % 5 > 0)
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    if not os.path.exists('gen_airplane/1'):
        os.makedirs('gen_airplane/1')
    vae.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for i in range(num):
            z = torch.randn(1, 64).to(device)  # Sample from the latent space
            generated_features = vae.decoder(z)
            generated_image = generated_features.view(3, 32, 32).cpu().numpy()  # Reshape and move to CPU
            plt.imsave(f'gen_airplane/1/generated_image_{i + 1}.png', np.transpose(generated_image, (1, 2, 0)))  # Save image
            axes[i // 5, i % 5].imshow(np.transpose(generated_image, (1, 2, 0)))  # Convert to HWC format for display
            axes[i // 5, i % 5].axis('off')
    plt.show()

def test_gen_img():
    B_ALEX = Alex_ee_inference.BranchedAlexNet(num_classes=10).to(device)
    B_ALEX.load_state_dict(torch.load(
        r"D:\Study\Module\Master Thesis\trained_models\B-Alex lr=0.001 transfer learning\B-Alex_cifar10_epoch_30.pth",
        weights_only=True))

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = datasets.ImageFolder(root='gen_airplane', transform=transform_test)
    dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    exit_thresholds = [0.5, 0.6, 0.3, 0.2, 0.2]
    accuracies, exit_ratios = Alex_ee_inference.threshold_inference(B_ALEX, dataloader, exit_thresholds)
    print(f'Accuracies: {accuracies}')
    print(f'Exit Ratios: {exit_ratios}')

if __name__ == '__main__':
    if TRAIN:
        train(1)
    generate_and_save(100)
    test_gen_img()

    # 100 samples result:
    # Accuracies: [100.0, 100.0, 80.0, 0.0, 0, 23.529411764705884]
    # Exit Ratios: [71.0, 6.0, 5.0, 1.0, 0.0, 17.0]
    # this means 78% are easy samples and classified correctly!!! very good result id say

