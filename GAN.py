import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16_bn
from CustomDataset import Data_prep_224_gen
import argparse


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels)
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#         out = self.block(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class Generator(nn.Module):
#     def __init__(self, latent_dim=256, init_channels=512):
#         super(Generator, self).__init__()
#
#         self.init_size = 7  # Initial size before upsampling
#
#         # Initial dense layer
#         self.init = nn.Sequential(
#             nn.Linear(latent_dim, init_channels * self.init_size * self.init_size),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#         # Reshape to convolutional features
#         self.conv_blocks = nn.Sequential(
#             # State size: (init_channels) x 7 x 7
#             nn.BatchNorm2d(init_channels),
#             ResidualBlock(init_channels),
#
#             # Upsampling block 1: 7x7 -> 14x14
#             nn.ConvTranspose2d(init_channels, init_channels//2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(init_channels//2),
#             nn.ReLU(inplace=True),
#             ResidualBlock(init_channels//2),
#
#             # Upsampling block 2: 14x14 -> 28x28
#             nn.ConvTranspose2d(init_channels//2, init_channels//4, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(init_channels//4),
#             nn.ReLU(inplace=True),
#             ResidualBlock(init_channels//4),
#
#             # Upsampling block 3: 28x28 -> 56x56
#             nn.ConvTranspose2d(init_channels//4, init_channels//8, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(init_channels//8),
#             nn.ReLU(inplace=True),
#             ResidualBlock(init_channels//8),
#
#             # Upsampling block 4: 56x56 -> 112x112
#             nn.ConvTranspose2d(init_channels//8, init_channels//16, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(init_channels//16),
#             nn.ReLU(inplace=True),
#             ResidualBlock(init_channels//16),
#
#             # Upsampling block 5: 112x112 -> 224x224
#             nn.ConvTranspose2d(init_channels//16, 3, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         # z: batch_size x latent_dim
#         out = self.init(z)
#         out = out.view(out.shape[0], 512, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        channel_att = torch.sigmoid(self.fc(avg) + self.fc(max_)).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att

class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3,
            stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=1, stride=2,
                output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.attention = CBAM(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = self.attention(F.relu(x))
        return x

class HighResGenerator(nn.Module):
    def __init__(self, z_dim=256, init_size=4):
        super().__init__()
        self.init_size = init_size
        self.init_channels = 512

        # Initial projection
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, self.init_channels * init_size ** 2),
            nn.BatchNorm1d(self.init_channels * init_size ** 2),
            nn.ReLU()
        )

        # Main blocks
        self.res_blocks = nn.Sequential(
            ResidualBlockUp(512, 256),  # 4x4 → 8x8
            ResidualBlockUp(256, 128),  # 8x8 → 16x16
            ResidualBlockUp(128, 64),  # 16x16 → 32x32
            ResidualBlockUp(64, 64),  # 32x32 → 64x64
            ResidualBlockUp(64, 64),  # 64x64 → 128x128
            ResidualBlockUp(64, 64),  # 128x128 → 256x256
        )

        # Final adjustment
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.l1(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.res_blocks(x)

        # 裁剪256x256到224x224
        h, w = x.size()[2:]
        dh = (h - 224) // 2
        dw = (w - 224) // 2
        x = x[:, :, dh:h - dh, dw:w - dw]

        return self.final_conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用原始VGG16-BN结构（输入224x224）
        self.backbone = vgg16_bn(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def initialize_model(all, category, latent_dim=256):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = HighResGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    if fine_tune:
        generator.load_state_dict(torch.load(''))
        discriminator.load_state_dict(torch.load(''))

    # Define loss and optimizers
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer_g = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.9))

    optimizer_g = optim.SGD(
        generator.parameters(),
        lr=0.005,  # Lower learning rate
        momentum=0.9,  # High momentum helps with stability
        weight_decay=1e-5,  # Small regularization
        nesterov=True  # Nesterov acceleration
    )

    optimizer_d = optim.SGD(
        discriminator.parameters(),
        lr=0.001,  # Even lower for discriminator to prevent it overpowering
        momentum=0.9,
        weight_decay=1e-5,
        nesterov=True
    )
    # get airplane/etc class index
    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_gen(root)

    if all == 0:
        train_idx, val_idx, test_idx = dataprep.get_category_index(category=category)  # 0 airplane, 3 cat, 8 ship
        # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
        train_loader, val_loader, test_loader = dataprep.create_catogery_loaders(batch_size=128, num_workers=2,
                                                                              train_idx=train_idx, val_idx=val_idx,
                                                                              test_idx=test_idx)
    elif all == 1:
        train_loader, val_loader, test_loader = dataprep.create_loaders(batch_size=128, num_workers=2)

    return generator, discriminator, criterion, optimizer_g, optimizer_d, train_loader, val_loader, test_loader, device, latent_dim

def train(num_epochs, all, category, save_interval=40):
    (generator, discriminator, criterion, optimizer_g, optimizer_d,
     train_loader, val_loader, test_loader, device, latent_dim) = initialize_model(all, category)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    G_val_losses = []
    D_val_losses = []

    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=50, eta_min=0.0001)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=50, eta_min=0.0001)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        total_d_loss = 0
        total_g_loss = 0
        total_batches = 0

        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            total_batches += 1

            # Labels with label smoothing
            real_label = torch.ones(batch_size, 1).to(device) * 0.9  # Smooth to 0.9
            fake_label = torch.zeros(batch_size, 1).to(device) + 0.1  # Smooth to 0.1

            #  Train Discriminator
            optimizer_d.zero_grad()

            # Loss on real images
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_label)

            # Loss on fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_label)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train generator less frequently (every 3 iterations)
            if i % 1 == 0:
                optimizer_g.zero_grad()

                # Generate new fake images
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(z)
                outputs = discriminator(fake_images)

                # Calculate generator loss
                g_loss = criterion(outputs, real_label)
                g_loss.backward()
                optimizer_g.step()

                total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_d_loss = total_d_loss / total_batches
        avg_g_loss = total_g_loss / total_batches

        # Save losses for plotting
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)

        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            torch.save(generator.state_dict(), f'training_weights/GAN/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'training_weights/GAN/discriminator_epoch_{epoch + 1}.pth')

            # Generate and save sample images
            with torch.no_grad():
                generator.eval()
                sample_z = torch.randn(4, latent_dim).to(device)
                sample_images = generator(sample_z)
                save_images(sample_images.cpu(), f'Results/GAN/training_samples/epoch_{epoch + 1}.png')
                generator.train()

        # Validation at the end of each epoch
        d_val_loss, g_val_loss = evaluate_model(generator, discriminator, criterion, val_loader, device, latent_dim)
        G_val_losses.append(g_val_loss)
        D_val_losses.append(d_val_loss)

        # Step the schedulers based on validation loss
        scheduler_g.step()
        scheduler_d.step()

        # Print progress
        print(f"train Epoch {epoch}/{num_epochs} "
              f"[D_loss: {avg_d_loss:.4f}] "
              f"[G loss: {avg_g_loss:.4f}]    "
              f"Val Epoch {epoch}/{num_epochs} "
              f"[D loss: {d_val_loss:.4f}] "
              f"[G loss: {g_val_loss:.4f}] " )


    # Plot both training and validation losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Training Losses")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Validation Losses")
    plt.plot(G_val_losses, label="Generator")
    plt.plot(D_val_losses, label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_losses.png')
    plt.close()

    return generator, discriminator

def test(generator_path, num_samples=25, latent_dim=256):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    generator = HighResGenerator(latent_dim=latent_dim).to(device)
    generator = nn.DataParallel(generator)
    generator.load_state_dict(torch.load(generator_path, weights_only=True))
    generator.eval()

    # Generate images
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_images = generator(z).cpu()

    # Display generated images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.flatten()

    for i, img in enumerate(generated_images):
        img = img / 2 + 0.5  # Unnormalize
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.show()

    return generated_images

def evaluate_model(generator, discriminator, criterion, val_loader, device, latent_dim):
    """Evaluate the model on validation data"""
    generator.eval()
    discriminator.eval()

    total_d_loss = 0
    total_g_loss = 0
    total_batches = 0

    with torch.no_grad():
        for real_images, _ in val_loader:
            batch_size = real_images.size(0)
            total_batches += 1
            real_images = real_images.to(device)

            # Labels with slight smoothing
            real_label = torch.ones(batch_size, 1).to(device) * 0.9
            fake_label = torch.zeros(batch_size, 1).to(device) + 0.1

            # Evaluate discriminator
            real_outputs = discriminator(real_images)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)

            # Calculate losses
            d_loss_real = criterion(real_outputs, real_label)
            d_loss_fake = criterion(fake_outputs, fake_label)
            d_loss = d_loss_real + d_loss_fake
            g_loss = criterion(fake_outputs, real_label)

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

    avg_d_loss = total_d_loss / total_batches
    avg_g_loss = total_g_loss / total_batches

    return avg_d_loss, avg_g_loss

def save_images(images, path):
    """Save a grid of generated images"""
    grid = torchvision.utils.make_grid(images, normalize=True)
    torchvision.utils.save_image(grid, path)

def generate_and_save_images(generator, latent_dim, device, num_images=100, output_path='generated_dataset.pth'):
    """Generate a dataset of images and save them"""
    generator.eval()
    generated_images = []

    with torch.no_grad():
        for i in range(0, num_images, 64):
            batch_size = min(64, num_images - i)
            z = torch.randn(batch_size, latent_dim).to(device)
            batch_images = generator(z).cpu()
            generated_images.append(batch_images)

    all_images = torch.cat(generated_images, 0)
    torch.save(all_images, output_path)
    print(f"Generated {len(all_images)} images and saved to {output_path}")

    return all_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'generate'])
    parser.add_argument('--category', type=int, default=0)
    parser.add_argument('--all', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--generator_path', type=str, default='weights/generator_epoch_50.pth')
    parser.add_argument('--num_samples', type=int, default=25)
    args = parser.parse_args()

    # Parameters you can adjust
    args.mode = 'train'  # Options: 'train', 'test', 'generate'
    fine_tune = False
    args.category = 0  # Category to generate (0 for airplane, 3 for cat, 8 for ship)
    args.all_data = 0  # 0 for specific category, 1 for all categories
    args.epochs = 200
    args.batch_size = 64
    args.latent_dim = 256
    args.num_samples = 5
    args.generator_path = 'training_weights/GAN/generator_epoch_50.pth'


    if args.mode == 'train':
        print("training mode activated")
        generator, discriminator = train(num_epochs=args.epochs, all = args.all, category = args.category)

    elif args.mode == 'test':
        print("test mode activated")
        test(args.generator_path, num_samples=args.num_samples, latent_dim=args.latent_dim)

    elif args.mode == 'generate':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = HighResGenerator(latent_dim=args.latent_dim).to(device)
        generator.load_state_dict(torch.load(args.generator_path, map_location=device))
        generate_and_save_images(generator, args.latent_dim, device, num_images=1000, output_path='generated_dataset.pth')
