import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
import Alex_ee_inference
import Vgg16_ee_inference
import Resnet50_ee
from Alexnet_early_exit import BranchedAlexNet
from Vgg16bn_early_exit_small_fc import BranchVGG16BN
from Resnet50_ee import BranchedResNet50
import matplotlib.pyplot as plt
from CustomDataset import Data_prep_224_gen
import datetime
from tqdm import tqdm
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset

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
        # Encoder: 224x224 → 112x112 → 56x56 → 28x28 → 14x14 → 7x7
        self.enc1 = nn.Sequential(  # 224 -> 112
            nn.Conv2d(3, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = nn.Sequential(  # 112 -> 56
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.enc3 = nn.Sequential(  # 56 -> 28
            nn.Conv2d(128, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.enc4 = nn.Sequential(  # 28 -> 14
            nn.Conv2d(256, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.enc5 = nn.Sequential(  # 14 -> 7
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # Decoder: 7x7 → 14x14 → 28x28 → 56x56 → 112x112 → 224x224
        self.dec1 = nn.Sequential(  # 7 -> 14
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
        )
        self.dec2 = nn.Sequential(  # 14 -> 28
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
        )
        self.dec3 = nn.Sequential(  # 28 -> 56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
        )
        self.dec4 = nn.Sequential(  # 56 -> 112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
        )
        self.dec5 = nn.Sequential(  # 112 -> 224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, 3, padding=1),
            # nn.Sigmoid()  # Output normalized to [0, 1]
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # 224 -> 112
        x2 = self.enc2(x1)  # 112 -> 56
        x3 = self.enc3(x2)  # 56 -> 28
        x4 = self.enc4(x3)  # 28 -> 14
        x5 = self.enc5(x4)  # 14 -> 7

        # Decoder with skip connections
        d1 = self.dec1(x5) + x4  # 7 -> 14
        d2 = self.dec2(d1) + x3  # 14 -> 28
        d3 = self.dec3(d2) + x2  # 28 -> 56
        d4 = self.dec4(d3) + x1  # 56 -> 112
        out = self.dec5(d4)  # 112 -> 224
        return out


def centered_l1_loss(pred, target, center_weight):
    # 创建中心加权的mask（例如：高斯权重）
    h, w = pred.shape[2], pred.shape[3]
    y = torch.linspace(-1, 1, h).view(h, 1).repeat(1, w)
    x = torch.linspace(-1, 1, w).view(1, w).repeat(h, 1)
    mask = torch.exp(-(x**2 + y**2) * center_weight)  # mask is (0,1]
    mask = mask.to(pred.device)
    return torch.mean(mask * torch.abs(pred - target))

def compute_loss(model, generated, original_img, target_label):
    # 早退损失
    if classifier_name == 'B_alex' or classifier_name == 'B_Vgg16':
        main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = classifier(generated)
        exit_confidences = compute_entropy_5(exit1_out, exit2_out, exit3_out, exit4_out, exit5_out)
    elif classifier_name == 'B_Resnet50':
        main_out, exit1_out, exit2_out, exit3_out, exit4_out = classifier(generated)
        exit_confidences = compute_entropy_4(exit1_out, exit2_out, exit3_out, exit4_out)

    exit_loss = torch.mean(exit_confidences)  # the mean of first exit's confidence, min it -> encourage early exit

    # 分类损失（最终层输出）

    # 1. labels as measurement
    # classified_label, exit_point = Alex_ee_inference.threshold_inference_new(model, 0, generated_224, [0.5, 0.6, 0.3, 0.2, 0.2])
    # cls_loss = nn.MSELoss()(classified_label.float(), target_label.float())

    # 2. use features as measurement
    gen_feature = classifier.module.extract_features(generated)
    ori_feature = classifier.module.extract_features(original_img)
    cls_loss1 = nn.MSELoss()(gen_feature, ori_feature)

    # cls_loss1 = nn.CrossEntropyLoss()(gen_feature, ori_feature) # label and exit point is great, but img is nothing

    # 3. use the end output logits and MSE
    cls_loss2 = nn.CrossEntropyLoss()(main_out, target_label)

    # 相似损失
    sim_loss = nn.L1Loss()(generated, original_img)
    # sim_loss = centered_l1_loss(generated, original_img, center_weight=5.0)

    return 500 * exit_loss  +  500 * cls_loss2 +  500 * cls_loss1 + 500 * sim_loss

def compute_entropy_5(exit1_out, exit2_out, exit3_out, exit4_out, exit5_out):
    softmax_exit1 = F.softmax(exit1_out, dim=1)
    entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

    softmax_exit2 = F.softmax(exit2_out, dim=1)
    entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

    softmax_exit3 = F.softmax(exit3_out, dim=1)
    entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

    softmax_exit4 = F.softmax(exit4_out, dim=1)
    entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

    softmax_exit5 = F.softmax(exit5_out, dim=1)
    entropy_exit5 = -torch.sum(softmax_exit5 * torch.log(softmax_exit5 + 1e-5), dim=1)

    return torch.stack([entropy_exit1, entropy_exit2, entropy_exit3, entropy_exit4, entropy_exit5], dim=0)

def compute_entropy_4(exit1_out, exit2_out, exit3_out, exit4_out):
    softmax_exit1 = F.softmax(exit1_out, dim=1)
    entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

    softmax_exit2 = F.softmax(exit2_out, dim=1)
    entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

    softmax_exit3 = F.softmax(exit3_out, dim=1)
    entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

    softmax_exit4 = F.softmax(exit4_out, dim=1)
    entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

    return torch.stack([entropy_exit1, entropy_exit2, entropy_exit3, entropy_exit4], dim=0)

def train(num_epoch):
    generator.train()
    loss_log_path = r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224/log.txt"
    epoch_loss = 0  # To accumulate loss for the entire epoch
    num_batches = 0 # To calculate average loss by dividing by # of batches

    for epoch in range(num_epoch):
        for idx, data in enumerate(trainloader):
            original_images, original_labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            generated_images = generator(original_images)
            loss = compute_loss(classifier, generated_images, original_images, original_labels)

            loss.backward()
            optimizer.step()

            # Accumulate the loss for the current batch
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epoch}, Training Loss: {avg_loss}")

        # Append the average loss to the local file
        with open(loss_log_path, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Epoch {epoch + 1}: {avg_loss}\n")

        if not os.path.exists(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224"):
                os.makedirs(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224")
        if (epoch+1) % 20 == 0:
            torch.save(generator.state_dict(),
                    f"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224/Generator_epoch_{epoch+1}.pth")

def test(batch, num):
    generator.eval()
    classifier.eval()
    num_forced_class_original = 0
    num_forced_class_generated = 0

    with torch.no_grad():
        for idx, data in enumerate(testloader):
            print(f"batch {idx+1}")
            if idx >= batch:
                break

            original_images, original_labels = data[0].to(device), data[1].to(device)
            original_images_crop = crop_img(original_images)

            if WHITE_BOARD_TEST:
                original_images.fill_(0)

            # print(original_images.max(), original_images.min(), original_images.mean())

            thresholds = [0.7, 0.8, 1.0, 0.8, 0.7]
            # thresholds = [0, 0.8, 1.0, 0.8, 0.7]
            classified_label, original_exit = Resnet50_ee.threshold_inference_new(classifier, 0, original_images,
                                                                                    thresholds)

            generated_images = generator(original_images)
            # print(generated_images.max(), generated_images.min(), generated_images.mean())
            generated_images_crop = crop_img(generated_images)

            #1 use diff to combine

            # diff = generated_images_crop - original_images
            # diff[:,:, only_edge: -only_edge, only_edge: -only_edge] = 0

            # diff[:, :, :only_edge, :] = 1
            # diff[:, :, -only_edge:, :] = 1
            # diff[:, :, :, :only_edge] = 1
            # diff[:, :, :, -only_edge:] = 1
            # generated_images_crop += 0.5*diff
            # generated_images_crop =  diff

            #2 directly combine ori and gen

            # original_images_copy = original_images.clone()
            # original_images_copy[:,:, only_edge: -only_edge, only_edge: -only_edge] = 2*generated_images_crop[:,:, only_edge: -only_edge, only_edge: -only_edge]
            # generated_images_crop = original_images_copy

            classified_label_gen, generated_exit = Resnet50_ee.threshold_inference_new(classifier, 0, generated_images_crop,
                                                                                      thresholds)

            print(f"Original Image - label: {classified_label}, \n Exit Location: {original_exit}")
            print(f"Generated Image - label: {classified_label_gen}, \n Exit Location: {generated_exit}")

            num_forced_class_original += (classified_label == category).sum().item()
            num_forced_class_generated += (classified_label_gen == category).sum().item()


            # Show the original and generated images
            for i in range(num): # show x imgs
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title(f"Original Image {i + 1}, "
                          f"label: {classified_label[i]}, "
                          f"exit: {original_exit[i]}")

                out1 = np.transpose(original_images[i].cpu().numpy(), (1, 2, 0))
                out1_normalized = (out1 - np.min(out1)) / (np.max(out1) - np.min(out1))

                plt.imshow(out1_normalized)
                plt.subplot(1, 2, 2)
                plt.title(f"crop size {crop_size}, replace size {replace_size}, only edge {only_edge}\n "
                          f"Gen Image {i + 1}, "
                          f"label: {classified_label_gen[i]}, "
                          f"exit: {generated_exit[i]}"
                )

                out2 = np.transpose(generated_images_crop[i].cpu().numpy(), (1, 2, 0))
                out2_normalized = (out2 - np.min(out2)) / (np.max(out2) - np.min(out2))
                plt.imshow(out2_normalized)
                plt.show()

        print(f"Number of {category}s in Original Labels: {num_forced_class_original}")
        print(f"Number of {category}s in Generated Labels: {num_forced_class_generated}")

def gen_dataset(generator, trainloader, valloader, testloader):

    def process_and_save_data(dataloader, save_path, dataset_name):

        generated_images = []
        labels = []

        print(f"Generating images for {dataset_name} dataset...")

        generator.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for images, batch_labels in tqdm(dataloader, desc=f"Processing {dataset_name}"):
                images = images.to(device)
                # Generate images
                batch_generated = generator(images)
                # Move data to CPU for saving
                generated_images.append(batch_generated.cpu())
                labels.append(batch_labels)

        # Concatenate tensors to create a single dataset
        generated_images = torch.cat(generated_images, dim=0)
        labels = torch.cat(labels, dim=0)

        # Save to .pkl file
        with open(save_path, "wb") as f:
            pickle.dump({"images": generated_images, "labels": labels}, f)

        print(f"Saved {dataset_name} dataset to {save_path}")

    # Process and save the training data
    process_and_save_data(trainloader, 'data_split/generated_CIFAR224_train.pkl', "generated_CIFAR224_train")

    # Process and save the validation data
    process_and_save_data(valloader, 'data_split/generated_CIFAR224_val.pkl', "generated_CIFAR224_val")

    process_and_save_data(testloader, 'data_split/generated_CIFAR224_test.pkl', "generated_CIFAR224_test")

def display_gen_dataset(trainset_path, valset_path, num_images=4):

    # Helper function to unnormalize and display an image
    def display_image(tensor, label, title_prefix, idx):

        # Convert the tensor to a numpy array and adjust it for display
        img = tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

        plt.imshow(img)
        plt.title(f"{title_prefix} Image {idx + 1}, Label: {label}")
        plt.axis("off")

    # Load the training and validation datasets
    with open(trainset_path, "rb") as f:
        train_data = pickle.load(f)
    with open(valset_path, "rb") as f:
        val_data = pickle.load(f)

    # Extract images and labels
    train_images, train_labels = train_data["images"], train_data["labels"]
    val_images, val_labels = val_data["images"], val_data["labels"]

    print(f"Loaded {len(train_images)} training images and {len(val_images)} validation images.")

    # Convert labels to NumPy arrays for easy random sampling
    train_labels = train_labels.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    plt.figure(figsize=(15, 6))

    # Random sampling of training images
    print(f"Displaying {num_images} random images from the training dataset...")
    sampled_train_indices = random.sample(range(len(train_images)), min(num_images, len(train_images)))
    for i, idx in enumerate(sampled_train_indices):
        plt.subplot(2, num_images, i + 1)
        display_image(train_images[idx], train_labels[idx], "Train", idx)

    # Random sampling of validation images
    print(f"Displaying {num_images} random images from the validation dataset...")
    sampled_val_indices = random.sample(range(len(val_images)), min(num_images, len(val_images)))
    for i, idx in enumerate(sampled_val_indices):
        plt.subplot(2, num_images, num_images + i + 1)
        display_image(val_images[idx], val_labels[idx], "Validation", idx)

    # Display all images
    plt.tight_layout()
    plt.show()

def crop_img(imgs):

    if crop_size != 0 and replace_size == 0 : # crop
        cropped_imgs = imgs[:, :, crop_size:-crop_size, crop_size:-crop_size]  # [B, C, 184, 184]
        final_imgs = torch.nn.functional.interpolate(cropped_imgs, size=(224, 224), mode="bilinear", align_corners=False)
    elif crop_size == 0 and replace_size != 0 : # replace
        processed_imgs = imgs.clone()
        processed_imgs[:, :, :replace_size, :] = 1  # Replace top edge
        processed_imgs[:, :, -replace_size:, :] = 1 # Replace bottom edge
        processed_imgs[:, :, :, :replace_size] = 1  # Replace left edge
        processed_imgs[:, :, :, -replace_size:] = 1  # Replace right edge
        final_imgs = processed_imgs
    # elif crop_size == 0 and replace_size == 0 and only_edge != 0:
    #     processed_imgs = imgs.clone()
    #     processed_imgs[:, :, only_edge:-only_edge, only_edge:-only_edge] = 0
    #     final_imgs =  processed_imgs

    else:
        final_imgs = imgs
        
    return final_imgs

def test_gen_dataset(testset_path, valset_path):
    generator.eval()
    classifier.eval()

    def load_data(file_path):
        # Load the .pkl file
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        images, labels = data["images"], data["labels"]
        dataset = TensorDataset(images, labels)  # Create a TensorDataset
        return dataset

    # Load the datasets
    print("Loading datasets...")
    test_dataset = load_data(testset_path)
    val_dataset = load_data(valset_path)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

    trainset_labels, trainset_exits = [], []
    valset_labels, valset_exits = [], []

    ori_trainset_labels, ori_trainset_exits = [], []
    ori_valset_labels, ori_valset_exits = [], []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            imgs, _ = data[0].to(device), data[1].to(device)
            cropped_imgs = crop_img(imgs)

            # Evaluate the model on the train and validation datasets
            trainset_label, trainset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, cropped_imgs,
                                                                                        [0.7, 0.8, 1.0, 0.8, 0.7])
            trainset_labels.append(trainset_label)
            trainset_exits.append(trainset_exit)

        for idx, data in enumerate(val_loader):
            imgs, _ = data[0].to(device), data[1].to(device)
            cropped_imgs = crop_img(imgs)

            valset_label, valset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, cropped_imgs,
                                                                                [0.7, 0.8, 1.0, 0.8, 0.7])
            valset_labels.append(valset_label)
            valset_exits.append(valset_exit)

        for idx, data in enumerate(testloader):
            imgs, _ = data[0].to(device), data[1].to(device)
            cropped_imgs = crop_img(imgs)
            ori_trainset_label, ori_trainset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, imgs,
                                                                                [0.7, 0.8, 1.0, 0.8, 0.7])

            ori_trainset_labels.append(ori_trainset_label)
            ori_trainset_exits.append(ori_trainset_exit)

        for idx, data in enumerate(valloader):
            imgs, _ = data[0].to(device), data[1].to(device)
            cropped_imgs = crop_img(imgs)
            ori_valset_label, ori_valset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, imgs,
                                                                            [0.7, 0.8, 1.0, 0.8, 0.7])

            ori_valset_labels.append(ori_valset_label)
            ori_valset_exits.append(ori_valset_exit)


    # print(f"Testset label: {trainset_labels}, \n Exit Location: {trainset_exits}")
    # print(f"Valset label: {valset_labels}, \n Exit Location: {valset_exits}")

    print(f"----Test Exit Location: {trainset_exits}")
    print(f'Original Exit Location: {ori_trainset_exits}')
    print(f"-----Val Exit Location: {valset_exits}")
    print(f'Original Exit Location: {ori_valset_exits}\n')

    diff = [
        [b_item - a_item for a_item, b_item in zip(a, b)]
        for a, b in zip(trainset_exits, ori_trainset_exits)
    ]
    negative_count = sum(
        1 for sublist in diff for value in sublist if value < 0
    )
    positive_count = sum(
        1 for sublist in diff for value in sublist if value > 0
    )
    total_diff = sum(len(sublist) for sublist in diff)

    print(f"early or late: {diff}")
    print(f"negative count: {negative_count} of {total_diff}")
    print(f"positive count: {positive_count} of {total_diff}")
    print(f"maintain count: {total_diff - positive_count - negative_count} of {total_diff}")

def initialize_model(classifier_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if classifier_name == 'B_alex':
        classifier = BranchedAlexNet()
        classifier.load_state_dict(
            torch.load(r"/home/yibo/PycharmProjects/Thesis/weights/B-Alex final/B-Alex_cifar10.pth", weights_only=True))

        classifier.to(device)
        classifier.eval()

    elif classifier_name == 'B_Vgg16':
        classifier = BranchVGG16BN()
        classifier.to(device)
        classifier = nn.DataParallel(classifier, device_ids=[0, 1, 2, 3])
        classifier.load_state_dict(torch.load(r"weights/Vgg16bn_ee_224/Vgg16bn_epoch_15.pth", weights_only=True))
        classifier.eval()

    elif classifier_name == 'B_Resnet50':
        classifier = BranchedResNet50()
        classifier.to(device)
        classifier = nn.DataParallel(classifier, device_ids=[0, 1, 2, 3])
        classifier.load_state_dict(torch.load(r"weights/Resnet50/B-Resnet50_epoch_10.pth", weights_only=True))
        classifier.eval()

    # get airplane/etc class index
    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_gen(root)
    train_idx, val_idx, test_idx = dataprep.get_category_index(category=category)  # 0 airplane, 3 cat, 8 ship
    # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
    trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=128, num_workers=2,
                                                                          train_idx=train_idx, val_idx=val_idx,
                                                                          test_idx=test_idx)

    generator = Generator().to(device)
    generator = nn.DataParallel(generator, device_ids=[0, 1, 2, 3])

    return generator, classifier, device, trainloader, valloader, testloader

if __name__ == "__main__":

    # B_alex Alex_ee_inference
    # B_Vgg16 Vgg16_ee_inference
    # B_Resnet50 Resnet50_ee
    pl.seed_everything(2024)

    crop_size = 0
    replace_size = 0
    only_edge = 0
    WHITE_BOARD_TEST = False
    TRAIN = False
    classifier_name = "B_Resnet50"
    category = 3 # 0 Airplane, 1 Automobile, 2 Bird, 3 Cat, 4 Deer, 5 Dog, 6 Frog, 7 Horse, 8 Ship, 9 Truck

    generator, classifier, device, trainloader, valloader, testloader = initialize_model(classifier_name)
    optimizer = optim.Adam(generator.parameters(), lr=1e-3)

    if TRAIN:
        train(200)
    else:
        # generator.load_state_dict(torch.load('weights/Generator224/vgg/Generator_epoch_50_ship.pth', weights_only=True))
        generator.load_state_dict(torch.load('training_weights/Generator224/Generator_epoch_60.pth', weights_only=True))

    test(5, 2)  # test n imgs per batch of testloader; in total 4n~5n imgs
    gen_dataset(generator, trainloader, valloader, testloader)
    # display_gen_dataset('data_split/generated_CIFAR224_test.pkl', 'data_split/generated_CIFAR224_val.pkl', 10)
    test_gen_dataset('data_split/generated_CIFAR224_test.pkl', 'data_split/generated_CIFAR224_val.pkl')

