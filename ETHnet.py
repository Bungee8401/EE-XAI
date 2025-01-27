import pytorch_lightning
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import Alex_ee_inference
import Alexnet_early_exit
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from CustomDataset import Data_prep_224_normal_N
import statistics

# Define the Ez_to_Hard_Net network
class ETH_Net(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(ETH_Net, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )
        # Define the decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()
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
        hard_img = self.decoder(z).view(-1, 3, 224, 224)
        return hard_img, mu, logvar


def eth_loss(original_acc, original_exit, new_acc, new_exit):

    if new_acc == 1:
        loss = math.exp( (new_exit - original_exit)) # exp(0.7x)
    else:
        print("wrong class generated!")
        loss = 30 * math.exp( (original_exit - new_exit))   # exp(0.7x)

    return torch.tensor(loss, requires_grad=True)

def vae_loss(recon_x, x, mu, logvar):

    # BCE = nn.BCEWithLogitsLoss(reduction='sum')(recon_x.view(-1, 224*224*3), x.view(-1, 224*224*3))
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + KLD
    return MSE + KLD

def vae_loss_1(recon_x, x, mu, logvar, new_exit, original_exit):
    if not (torch.all(recon_x >= 0) and torch.all(recon_x <= 1)):
        print(recon_x)
        recon_x = torch.clamp(recon_x, 0, 1)
        print("recon_x is out of range [0, 1], and clamped to [0, 1]")
        print(recon_x)
    if not (torch.all(x >= 0) and torch.all(x <= 1)):
        print(x)
        x = torch.clamp(x, 0, 1)
        print("x is out of range [0, 1], and clamped to [0, 1]")
        print(x)
        # raise ValueError("x is out of range [0, 1]")

    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 224 * 224 * 3), x.view(-1, 224 * 224 * 3),
                                             reduction='sum')  # how well decoder reconstructs
    # BCE = nn.functional.binary_cross_entropy_with_logits(recon_x.view(-1, 224*224*3), x.view(-1, 224*224*3), reduction='sum')  # how well decoder reconstructs
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(
        2) - logvar.exp())  # how closely the latent variable distribution matches the target distribution

    original_exit = torch.tensor(original_exit, dtype=torch.float32)
    new_exit = torch.tensor(new_exit, dtype=torch.float32)

    # todo: acc should also take into count
    if new_exit <= original_exit:
        exit_penalty = -500 * torch.exp(original_exit - new_exit)  # Reward for early exit
    else:
        exit_penalty = 500 * torch.exp(new_exit - original_exit)  # Penalty for later exit

    return BCE + KLD + exit_penalty

def acc_and_exit(data):

    exit_thresholds = [0.5, 0.6, 0.3, 0.2, 0.2]
    accuracy, exit_position = Alex_ee_inference.threshold_1_inference(B_ALEX, data, exit_thresholds) # model, dataloader, exit_thresholds

    # print(f'wrong class gen!') if accuracy == 0 else None
    # print(f'Exit Position: {exit_position}')

    return accuracy, exit_position

def test_n_samples(n):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #will have clipping issue in imshow()
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=r'D:\Code\Thesis\Airplane_right\exit3', transform=transform_train)
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    ETH_model.eval()
    with torch.no_grad():
        for i, (ez_input, label) in enumerate(dataloader):
            if i >= n:
                break
            ez_input = ez_input.to(device)
            hard_out, _, _ = ETH_model(ez_input)

            original_acc, original_exit = acc_and_exit(ez_input)
            new_acc, new_exit = acc_and_exit(hard_out)

            # Display input and output images with labels
            ez_input_img = ez_input.cpu().numpy().transpose(0, 2, 3, 1)[0]
            # hard_out_img = hard_out.squeeze().view(224, 224, 3).cpu().numpy()
            hard_out_img = hard_out.cpu().numpy().transpose(0, 2, 3, 1)[0]

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(ez_input_img)
            axes[0].set_title(f'Input Image exit at {original_exit}, acc is {original_acc}')
            axes[0].axis('off')

            axes[1].imshow(hard_out_img)
            axes[1].set_title(f'Output Image exit at {new_exit}, acc is {new_acc}')
            axes[1].axis('off')

            plt.show()

    pass

def train(num_epochs, train_loader, val_loader):

    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    #     todo: for BCE loss, N(0,1); but current dataloader normalize is not
    # ])

    train_losses = []
    # val_losses = []

    for epoch in range(num_epochs):
        ETH_model.train()
        train_loss = 0
        for batch_idx, (ez_input, _) in enumerate(train_loader):
            ez_input = ez_input.to(device)

            optimizer.zero_grad()

            hard_out, mu, logvar = ETH_model(ez_input)

            # original_acc, original_exit = acc_and_exit(ez_input)
            # new_acc, new_exit = acc_and_exit(hard_out)

            # loss = vae_loss_1(hard_out, ez_input, mu, logvar, new_exit, original_exit)
            loss = vae_loss(hard_out, ez_input, mu, logvar)
            # loss = eth_loss(original_acc, original_exit, new_acc, new_exit)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_losses.append(loss.item())

            # print(len(train_loader))
        print(f'Epoch {epoch + 1}, train Loss: {train_loss/len(train_loader)}')

        # # Validation
        # ETH_model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for batch_idx, (ez_input, _) in enumerate(val_loader):
        #         ez_input = ez_input.to(device)
        #
        #         hard_out, mu, logvar = ETH_model(ez_input)
        #
        #         # original_acc, original_exit = acc_and_exit(ez_input)
        #         # new_acc, new_exit = acc_and_exit(hard_out)
        #
        #         # loss = vae_loss_1(hard_out, ez_input, mu, logvar, new_exit, original_exit)
        #         loss = vae_loss(hard_out, ez_input, mu, logvar)
        #         # loss = eth_loss(original_acc, original_exit, new_acc, new_exit)
        #
        #         val_loss += loss.item()
        #         val_losses.append(loss.item())
        #
        # print(f'Epoch {epoch + 1}, val Loss: {val_loss / len(val_loader.dataset)}')

        if (epoch + 1) % 10 == 0:
            torch.save(ETH_model.state_dict(), f'ETH_Net_epoch_{epoch + 1}.pth')

    return train_losses

def initial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_lightning.seed_everything(2024)

    # model initialization
    ETH_model = ETH_Net(feature_dim=224 * 224 * 3, latent_dim=64).to(device)
    if os.path.exists('ETH_airplane.pth'):
        ETH_model.load_state_dict(torch.load('ETH_airplane.pth'))
        TRAIN = False
    else:
        TRAIN = True

    B_ALEX = Alex_ee_inference.BranchedAlexNet(num_classes=10).to(device)
    B_ALEX.load_state_dict(torch.load(r"D:\Study\Module\Master Thesis\trained_models\B-Alex final\B-Alex_cifar10.pth", weights_only=True))

    optimizer = optim.Adam(ETH_model.parameters(), lr=1e-3)

    return ETH_model, TRAIN, device, B_ALEX, optimizer


if __name__ == '__main__':

    ETH_model, TRAIN, device, B_ALEX, optimizer = initial()

    # get airplane/etc class index
    root = 'D:/Study/Module/Master Thesis/dataset/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    train_idx, val_idx, test_idx = dataprep.get_category_index(category = 0) # airplane
    # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
    trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=16, num_workers=2, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    if TRAIN:
        num_epochs = 30
        train_losses, val_losses = train(num_epochs, trainloader, valloader)

    print("Testing...")
    # test_n_samples(10)


    # for idx, (data, target) in enumerate(trainloader):
    #     data, target = data.to(device), target.to(device)
    #     # print(data.shape)
    #     print("-----------------")
    #     print(target)
    #     print(target.shape)
    #     break
    # for idx, (data, target) in enumerate(valloader):
    #     data, target = data.to(device), target.to(device)
    #     # print(data.shape)
    #     print("-----------------")
    #     print(target)
    #     print(target.shape)
    #     break
    # for idx, (data, target) in enumerate(testloader):
    #     data, target = data.to(device), target.to(device)
    #     # print(data.shape)
    #     print("-----------------")
    #     print(target)
    #     print(target.shape)
    #     break