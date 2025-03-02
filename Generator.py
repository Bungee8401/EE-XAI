import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import math
import torch.optim as optim
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import Alex_ee_inference
import Alexnet_early_exit
from Alexnet_early_exit import BranchedAlexNet
import matplotlib.pyplot as plt
from CustomDataset import Data_prep_32_N

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 编码器（Encoder）
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#         )
#         self.pool1 = nn.MaxPool2d(2)  # 32x32 → 16x16
#
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(),
#         )
#         self.pool2 = nn.MaxPool2d(2)  # 16x16 → 8x8
#
#         # 解码器（Decoder）
#         self.dec1 = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),  # 8x8 → 16x16
#             nn.ReLU(),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.ReLU(),
#         )
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding=1),  # 输入通道数64 → 32
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),  # 上采样到32x32
#             nn.ReLU(),
#             nn.Conv2d(32, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
#
#         # 跳跃连接的通道调整
#         self.skip_conv = nn.Conv2d(32, 32, 1)  # 调整编码器特征的通道数
#
#     def forward(self, x):
#         # 编码器
#         x1 = self.enc1(x)  # [B, 32, 32, 32]
#         x1_pooled = self.pool1(x1)  # [B, 32, 16, 16]
#
#         x2 = self.enc2(x1_pooled)  # [B, 64, 16, 16]
#         x2_pooled = self.pool2(x2)  # [B, 64, 8, 8]
#
#         # 解码器
#         d1 = self.dec1(x2_pooled)  # [B, 32, 16, 16]
#
#         # 跳跃连接：调整编码器特征的通道数并拼接
#         x1_skip = self.skip_conv(x1_pooled)  # [B, 32, 16, 16]
#         d1 = torch.cat([d1, x1_skip], dim=1)  # [B, 32+32=64, 16, 16]
#
#         # 最终上采样
#         output = self.dec2(d1)  # [B, 3, 32, 32]
#         return output


# In decoder blocks:
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)  # Residual connection

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Downsample
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # Downsample
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Upsample 8x8 → 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            ResidualBlock(128),

            # Upsample 16x16 → 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            ResidualBlock(64),

            # Final output
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # Fixed skip connections
        self.skip1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),  # Match decoder's 64 channels
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x1 = self.enc1(x)  # [B, 64, 16, 16]
        x2 = self.enc2(x1)  # [B, 128, 8, 8]

        # Decode with skip connections
        d1 = self.decoder[0:3](x2) + self.skip2(x2)  # 16x16
        d2 = self.decoder[3:6](d1) + self.skip1(x1)  # 32x32
        out = self.decoder[6:](d2)
        return out

def compute_loss(model, generated, original_img, target_label):
    # 早退损失（假设第一个退出点的置信度）
    generated_224 = F.interpolate(
        generated,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    original_224 = F.interpolate(
        original_img,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = B_alex(generated_224)
    exit_confidences = compute_entropy(exit1_out)
    exit_loss = torch.mean(exit_confidences) # cal the mean of first exit's confidence, min it -> encourage early exit

    # 分类损失（最终层输出）

    # 1. labels as measurement
    # classified_label, exit_point = Alex_ee_inference.threshold_inference_new(model, 0, generated_224, [0.5, 0.6, 0.3, 0.2, 0.2])
    # cls_loss = nn.MSELoss()(classified_label.float(), target_label.float())

    # 2. use features as measurement
    gen_feature = B_alex.extract_features(generated_224)
    ori_feature = B_alex.extract_features(original_224)
    cls_loss1 = nn.MSELoss()(gen_feature, ori_feature)

    # cls_loss1 = nn.CrossEntropyLoss()(gen_feature, ori_feature) # label and exit point is great, but img is nothing

    # 3. use the end output logits and MSE
    cls_loss2 = nn.CrossEntropyLoss()(main_out, target_label)

    # 相似损失
    sim_loss = nn.L1Loss()(generated, original_img)

    return 300*exit_loss  +  500 * cls_loss2 +  500 * cls_loss1 + 500 * sim_loss

def compute_entropy(exit1_out):
    softmax_exit1 = F.softmax(exit1_out, dim=1)
    entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

    # model.eval()
    # # entropy-based criteria as in BranchyNet; in cifar10, max entropy is 2.3; smaller entropy, more confident
    # with torch.no_grad():
    #     for data in dataloader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #
    #         # Forward pass
    #         main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)
    #
    #         # Calculate softmax and entropy for exit1-5
    #         softmax_exit1 = F.softmax(exit1_out, dim=1)
    #         entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)
    #
    #         softmax_exit2 = F.softmax(exit2_out, dim=1)
    #         entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)
    #
    #         softmax_exit3 = F.softmax(exit3_out, dim=1)
    #         entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)
    #
    #         softmax_exit4 = F.softmax(exit4_out, dim=1)
    #         entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)
    #
    #         softmax_exit5 = F.softmax(exit5_out, dim=1)
    #         entropy_exit5 = -torch.sum(softmax_exit5 * torch.log(softmax_exit5 + 1e-5), dim=1)


    return entropy_exit1

def train(num_epoch):
    generator.train()
    loss_log_path = r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator/log.txt"
    epoch_loss = 0  # To accumulate loss for the entire epoch
    num_batches = 0 # To calculate average loss by dividing by # of batches

    for epoch in range(num_epoch):
        for idx, data in enumerate(trainloader):
            original_images, original_labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            generated_images = generator(original_images)
            loss = compute_loss(B_alex, generated_images, original_images, original_labels)

            loss.backward()
            optimizer.step()

            # Accumulate the loss for the current batch
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epoch}, Training Loss: {avg_loss}")

        # Append the average loss to the local file
        with open(loss_log_path, "a") as f:
            f.write(f"Epoch {epoch + 1}: {avg_loss}\n")

        if not os.path.exists(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator"):
                os.makedirs(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator")
        if (epoch+1) % 50 == 0:
            torch.save(generator.state_dict(),
                    f"/home/yibo/PycharmProjects/Thesis/training_weights/Generator/Generator_epoch_{epoch+1}.pth")

def test(num):
    generator.eval()
    B_alex.eval()

    with torch.no_grad():
        for idx, data in enumerate(testloader):
            print(f"batch {idx+1}")
            if idx >= num:
                break

            original_images, original_labels = data[0].to(device), data[1].to(device)
            original_images_224 = F.interpolate(
                original_images,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

            # Step 2: Give samples to B_alex, threshold_inference_new to see its acc and exit location
            classified_label, original_exit = Alex_ee_inference.threshold_inference_new(B_alex, 0, original_images_224,
                                                                                    [0.7, 0.8, 1.0, 0.8, 0.7])

            # Step 3: Give samples to generator, get the generated imgs
            generated_images_32 = generator(original_images)
            generated_images_224 = F.interpolate(
                generated_images_32,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
            # print(generated_images_32.dtype)
            # print(original_images.dtype)

            # Step 4: Feed the img again to threshold_inference_new to see its acc and exit location
            classified_label_gen, generated_exit = Alex_ee_inference.threshold_inference_new(B_alex, 0, generated_images_224,
                                                                                      [0.0, 0.8, 1.0, 0.8, 0.7]) # change here to disable a certain exit location

            # Step 5: Print the accs and exit locations, also show the two imgs
            print(f"Original Image - label: {classified_label}, \n Exit Location: {original_exit}")
            print(f"Generated Image - label: {classified_label_gen}, \n Exit Location: {generated_exit}")

            # Show the original and generated images
            for i in range(1): # show x imgs
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title(f"Original Image {i + 1}, "
                          f"label: {classified_label[i]}, "
                          f"exit: {original_exit[i]}")

                out1 = np.transpose(original_images[i].cpu().numpy(), (1, 2, 0))
                out1_normalized = (out1 - np.min(out1)) / (np.max(out1) - np.min(out1))

                plt.imshow(out1_normalized)
                plt.subplot(1, 2, 2)
                plt.title(f"Generated Image {i + 1}, "
                          f"label: {classified_label_gen[i]}, "
                          f"exit: {generated_exit[i]}")

                out2 = np.transpose(generated_images_32[i].cpu().numpy(), (1, 2, 0))
                out2_normalized = (out2 - np.min(out2)) / (np.max(out2) - np.min(out2))
                plt.imshow(out2_normalized)
                plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(2024)

    generator = Generator().to(device)
    generator = nn.DataParallel(generator)

    B_alex = BranchedAlexNet()
    B_alex.load_state_dict(torch.load(r"/home/yibo/PycharmProjects/Thesis/weights/B-Alex final/B-Alex_cifar10.pth", weights_only=True))
    B_alex.to(device)
    B_alex = nn.DataParallel(B_alex)

    # get airplane/etc class index
    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_32_N(root)
    train_idx, val_idx, test_idx = dataprep.get_category_index(category = 5) # 0 airplane, 1....
    # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
    trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=256, num_workers=8, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    optimizer = optim.Adam(generator.parameters(), lr=1e-4)

    generator.load_state_dict(torch.load('/home/yibo/PycharmProjects/Thesis/weights/Generator/'
                                         'Generator_epoch_300_dog.pth', weights_only=True))

    # train(200)
    test(10) # test n batch of testloader


