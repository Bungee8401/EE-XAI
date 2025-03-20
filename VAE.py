import pytorch_lightning
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from CustomDataset import Data_prep_224_01_N
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Sigmoid()  # For normalized images in [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)

        # Special initialization for variational parts to prevent NaN
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20, max=20)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 512, 7, 7)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def norm_img(img):
    img = img - img.min()
    img = img / img.max()
    return img

def vae_loss_function(recon_x, x, mu, logvar, kld_weight):
    """
    Combined loss for VAE: reconstruction loss + KL divergence
    """
    # Use MSE loss for better quality with natural images
    recon_loss1 = F.mse_loss(recon_x, x, reduction='sum')
    # recon_loss2 = F.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = recon_loss1 #+ recon_loss2

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Safeguard against NaN
    if torch.isnan(kld_loss):
        print("kld_loss is nan, set to 0")
        kld_loss = torch.tensor(0.0).to(x.device)

    # Total loss
    loss = recon_loss + kld_weight * kld_loss

    return loss, recon_loss, kld_loss

def generate_from_sampling(num):
    rows = num // 5 + (num % 5 > 0)
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    # Make axes 2D if there's only one row
    if rows == 1:
        axes = axes.reshape(1, -1)

    vae.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for i in range(num):
            z = torch.randn(1, 256).to(device)  # Sample from the latent space
            recon_img = vae.module.decode(z)

            # Correctly reshape: [batch, channels, height, width] → [height, width, channels]
            recon_img = recon_img[0].permute(1, 2, 0).cpu().numpy()

            recon_img = norm_img (recon_img)

            # Save the image
            # plt.imsave(f'gen_airplane/from_sampling/generated_{i + 1}.png', recon_img)

            # Display in plot
            axes[i // 5, i % 5].imshow(recon_img)
            axes[i // 5, i % 5].axis('off')

    plt.tight_layout()
    # plt.savefig('generated_samples.png')
    plt.show()

def generate_and_save_from_input(num):
    rows = num // 5 + (num % 5 > 0)
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    if not os.path.exists('gen_airplane/from_input/1'):
        os.makedirs('gen_airplane/from_input/1')

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # for BCE loss
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=r'D:\Code\Thesis\Airplane_right\exit1', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    vae.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            if i >= num:
                break
            data = data.to(device)
            recon_batch, mu, logvar = vae(data)
            # z = torch.randn(1, 64).to(device)  # Sample from the latent space
            # generated_features = vae.decoder(z)

            generated_image = recon_batch.view(3, 32, 32).cpu().numpy()  # Reshape and move to CPU
            plt.imsave(f'gen_airplane/from_input/1/generated_image_{i + 1}.png', np.transpose(generated_image, (1, 2, 0)))  # Save image
            axes[i // 5, i % 5].imshow(np.transpose(generated_image, (1, 2, 0)))  # Convert to HWC format for display
            axes[i // 5, i % 5].axis('off')
    plt.show()

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 224*224*3), reduction='sum')
    # MSE = nn.functional.mse_loss(recon_x, x.view(-1, 32 * 32 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
    # return MSE + KLD

def display_imgs_in_training(epoch, valloader, num):
    vae.eval()
    with torch.no_grad():
        # Get a batch of validation data
        val_data, _ = next(iter(valloader))
        val_data = val_data.to(device)

        # Get reconstructions
        recon_batch, _, _ = vae(val_data)

        # Display original and reconstructed images
        fig, axes = plt.subplots(2, num, figsize=(12, 4))

        for i in range(num):
            # Original images
            orig_img = val_data[i].cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            # Reconstructed images
            recon_img = recon_batch[i].cpu().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title('Recon')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(f'Results/VAE/cat/reconstruction_epoch_{epoch}.png')
        plt.close()

    vae.train()

def train(num_epochs, trainloader, valloader, kld_weight):
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss_total = 0
        kld_loss_total = 0

        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            loss, recon_loss, kld_loss = vae_loss_function(recon_batch, data, mu, logvar, kld_weight)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kld_loss_total += kld_loss.item()

        # Adjust KLD weight (optional)
        kld_weight = min(kld_weight + 0.01, 1.0)  # Gradually increase to 1.0

        print(f'Epoch: {epoch}/{num_epochs}, total Loss: {total_loss:.4f}, '
              f'Recon Loss: {recon_loss_total:.4f}, '
              f'KLD: {kld_loss_total:.4f}')

        # Visualize reconstructions every few epochs
        if epoch % 20 == 0:
            vae.eval()  # Set to evaluation mode for visualization
            display_imgs_in_training(epoch, valloader, num = 5)
            vae.train()  # Set back to training mode

        if (epoch + 1) % 50 == 0:
            torch.save(vae.state_dict(), f'training_weights/VAE/VAE_{epoch + 1}.pth')


if __name__ == '__main__':
    pytorch_lightning.seed_everything(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE().to(device)
    vae = nn.DataParallel(vae)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)  # todo: SGD somehow dont work here

    TRAIN = False
    kld_weight = 0.001

    # get airplane/etc class index
    root = 'CIFAR10'
    dataprep = Data_prep_224_01_N(root)
    train_idx, val_idx, test_idx = dataprep.get_category_index(category=3)
    trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=128, num_workers=2,
                                                                          train_idx=train_idx, val_idx=val_idx,
                                                                          test_idx=test_idx)

    if TRAIN:
        vae.load_state_dict(torch.load('training_weights/VAE/VAE_500.pth', weights_only= True))
        train(500, trainloader, valloader, kld_weight)
    else:
        vae.load_state_dict(torch.load('training_weights/VAE/VAE_500.pth', weights_only= True))
        generate_from_sampling(20)
        # generate_and_save_from_input(20)


    # todo at 06 Jan
    #       1. use exit 1-3 to train
    #       2. use 224*224 size to train and inference
    #       3. ? maybe the latent_dim parameter?
    #       4. think, how to use this VAE to modify the hard samples

    # todo at 20 Jan
    #       1. why this file works before in 32*32? because local dataset is 224? because 300 epochs?
    #       2. 32*32 could be better in samller dataset size
    #       3. 32*32 may be bad. try 224*224
    #       4. 50~100 epoch is enough for 32*32