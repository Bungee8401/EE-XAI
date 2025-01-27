import pytorch_lightning
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import Alex_ee_inference
from CustomDataset import Data_prep_224_01_N

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


def seed(num):
    pytorch_lightning.seed_everything(num)

def generate_and_save_from_sampling(num):
    rows = num // 5 + (num % 5 > 0)
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    if not os.path.exists('gen_airplane/from_sampling/1'):
        os.makedirs('gen_airplane/from_sampling/1')
    vae.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for i in range(num):
            z = torch.randn(1, 128).to(device)  # Sample from the latent space
            generated_features = vae.decoder(z)
            generated_image = generated_features.view(3, 224, 224).cpu().numpy()  # Reshape and move to CPU
            plt.imsave(f'gen_airplane/from_sampling/1/generated_image_{i + 1}.png', np.transpose(generated_image, (1, 2, 0)))  # Save image
            axes[i // 5, i % 5].imshow(np.transpose(generated_image, (1, 2, 0)))  # Convert to HWC format for display
            axes[i // 5, i % 5].axis('off')
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

def train(num_epochs, trainloader, valloader):
    # # Define transforms for data augmentation and normalization
    # transform_train = transforms.Compose([
    #     transforms.Resize((32, 32)),  # Resize to 32x32
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # for BCE loss
    # ])

    # # Load the dataset
    # train_dataset = datasets.ImageFolder(root=r'D:\Code\Thesis\Airplane_right\exit1', transform=transform_train)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(valloader):
                data = data.to(device)
                recon_batch, mu, logvar = vae(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}, Trainning Loss: {train_loss / len(trainloader)}')
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(valloader)}')

        if (epoch + 1) % 50 == 0:
            torch.save(vae.state_dict(), f'VAE_airplane_{epoch + 1}.pth')

def test_gen_img():
    B_ALEX = Alex_ee_inference.BranchedAlexNet(num_classes=10).to(device)
    B_ALEX.load_state_dict(torch.load(
        r"D:\Study\Module\Master Thesis\trained_models\B-Alex final\B-Alex_cifar10.pth",
        weights_only=True))

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = datasets.ImageFolder(root='gen_airplane/from_sampling', transform=transform_test)
    dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    exit_thresholds = [0.5, 0.6, 0.3, 0.2, 0.2]
    accuracies, exit_ratios = Alex_ee_inference.threshold_inference(B_ALEX, dataloader, exit_thresholds)
    print(f'Accuracies: {accuracies}')
    print(f'Exit Ratios: {exit_ratios}')


if __name__ == '__main__':
    seed(2024)
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize VAE model
    vae = VAE(feature_dim=224 * 224 * 3, latent_dim=128).to(device)

    if os.path.exists('VAE_airplane_100.pth'):
        vae.load_state_dict(torch.load('VAE_airplane_100.pth'))
        TRAIN = False
    else:
        TRAIN = True

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)  # todo: SGD somehow dont work here

    # get airplane/etc class index
    root = 'D:/Study/Module/Master Thesis/dataset/CIFAR10'
    dataprep = Data_prep_224_01_N(root)
    train_idx, val_idx, test_idx = dataprep.get_category_index(category=0)  # airplane
    # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
    trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=100, num_workers=2,
                                                                          train_idx=train_idx, val_idx=val_idx,
                                                                          test_idx=test_idx)
    if TRAIN:
        train(500, trainloader, valloader)
    # train(500, trainloader, valloader)

    generate_and_save_from_sampling(20)
    test_gen_img()

    # plot some images from trainloader
    # data_iter = iter(trainloader)
    # images, labels = next(data_iter)
    # print(images.shape, labels)
    # fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    # for i in range(5):
    #     image = images[i].numpy().transpose((1, 2, 0))
    #     axes[i].imshow(image)
    #     axes[i].axis('off')
    # plt.show()

    # 100 samples result:
    # Accuracies: [100.0, 100.0, 80.0, 0.0, 0, 23.529411764705884]
    # Exit Ratios: [71.0, 6.0, 5.0, 1.0, 0.0, 17.0]
    # this means 78% are easy samples and classified correctly!!! very good result id say

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