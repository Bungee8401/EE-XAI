import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
import torchvision
import torchvision.models as models
import Resnet50_ee
from Alexnet_early_exit import BranchedAlexNet
from Vgg16bn_early_exit_small_fc import BranchVGG16BN
from Resnet50_ee import BranchedResNet50
import matplotlib.pyplot as plt
from CustomDataset import Data_prep_224_gen, load_cifar100_generator, create_cifar100_superclass_dataloaders
import datetime
from tqdm import tqdm
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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

class ResNet_50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_50, self).__init__()
        self.num_classes = num_classes

        # Load the pretrained ResNet50 model
        resnet = models.resnet50(weights=None)

        # Extract layers from the pretrained ResNet50 model
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Main classifier
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        out_main = self.fc(x)

        return out_main

    def extract_features(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            return x


def img_norm(img):
    min_val = torch.min(img)
    max_val = torch.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

def load_data(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

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
    # classified_label, exit_point = Resnet50_ee.threshold_inference_new(model, 0, generated_224, [0.5, 0.6, 0.3, 0.2, 0.2])
    # cls_loss = nn.MSELoss()(classified_label.float(), target_label.float())

    # 2. use features as measurement
    if classifier_name == 'B_alex':
        gen_feature = classifier.extract_features(generated)
        ori_feature = classifier.extract_features(original_img)
    if classifier_name == 'B_Vgg16' or classifier_name == 'B_Resnet50':
        gen_feature = classifier.extract_features(generated)
        ori_feature = classifier.extract_features(original_img)

    cls_loss1 = nn.MSELoss()(gen_feature, ori_feature)

    # cls_loss1 = nn.CrossEntropyLoss()(gen_feature, ori_feature) # label and exit point is great, but img is nothing

    # 3. use the end output logits and MSE
    cls_loss2 = nn.CrossEntropyLoss()(main_out, target_label)

    # 相似损失
    sim_loss = nn.L1Loss()(generated, original_img)
    # sim_loss = centered_l1_loss(generated, original_img, center_weight=5.0)

    return EE_LOSS_PARA * exit_loss  +  500 * cls_loss2 +  500 * cls_loss1 + 500 * sim_loss

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

def grad_cam_mask(label, images, resnet_for_cam):

    # 1. captum
    # layer_gradcam = LayerGradCam(resnet_for_cam, resnet_for_cam.module.layer4[2].conv3) # Resnet50
    # attributions_lgc = layer_gradcam.attribute(images, target=label)
    # upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, images.shape[2:])
    # return upsamp_attr_lgc

    # 2. pytorch_grad_cam
    cam = GradCAM(model=resnet_for_cam, target_layers=[resnet_for_cam.module.layer4[-1]])
    masked_images = torch.zeros_like(images).to(torch.uint8)
    for idx in range(images.shape[0]):
        input_tensor = images[idx].unsqueeze(0)  # Add batch dimension back
        target = ClassifierOutputTarget(label[idx].item())

        # Generate GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
        grayscale_cam = torch.from_numpy(grayscale_cam[0]).to(device)  # Convert to tensor and move to GPU

        # Apply GradCAM mask directly to the original image

        masked_img = img_norm(input_tensor).squeeze(0) * grayscale_cam.unsqueeze(0)  # Add channel dimension to mask

        # Normalize the masked image
        masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min())

        # Convert normalized float values [0,1] to integers [0,255]
        masked_img = (masked_img * 255).to(torch.uint8)

        # Store in output tensor
        masked_images[idx] = masked_img

        # if idx < 10:
        #     plt.figure(figsize=(15, 5))
        #
        #     # Original image
        #     plt.subplot(1, 3, 1)
        #     orig_img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        #     orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        #     plt.imshow(orig_img)
        #     plt.title(f'Original Image {idx}')
        #
        #     # GradCAM mask
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(grayscale_cam.cpu().numpy(), cmap='jet')
        #     plt.title(f'GradCAM Mask {idx}')
        #
        #     # Masked image
        #     plt.subplot(1, 3, 3)
        #     masked_viz = masked_img.cpu().permute(1, 2, 0).numpy()
        #     plt.imshow(masked_viz)
        #     plt.title(f'Masked Image {idx}')
        #
        #     plt.show()
    return masked_images

def train(num_epoch):
    generator.train()
    loss_log_path = r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator_CNN/log.txt"
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

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epoch}, Training Loss: {avg_loss}")

        # Append the average loss to the local file
        with open(loss_log_path, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Epoch {epoch + 1}: {avg_loss}\n")

        if not os.path.exists(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator_CNN"):
                os.makedirs(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator_CNN")
        if (epoch+1) % 10 == 0:
            torch.save({
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'training_weights/Generator_CNN/Generator_epoch_{epoch+1}.pth')
            white_board_test()
            test(10,1)

    print(f"Training complete for class {category}.")

def white_board_test():
    generator.eval()
    classifier.eval()

    # Create blank (white) images
    blank_images = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        # Generate images from blank input
        generated_images = generator(blank_images)

        # Save the generated image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'Results/white_board/{data_type}/{classifier_name}'
        os.makedirs(save_dir, exist_ok=True)

        # Get prediction from classifier
        classified_label_gen, generated_exit = Resnet50_ee.threshold_inference_new(classifier, 0, generated_images, thresholds)

        torchvision.utils.save_image(generated_images[0], f'{save_dir}/LossPara_{EE_LOSS_PARA}_Label_{category}:{classified_label_gen.item()}_EE_{generated_exit}_time_{timestamp}.png',
                                    normalize=True)

def test(batch, num):
    generator.eval()
    classifier.eval()
    num_correct_class_original = 0
    num_correct_class_generated = 0

    save_dir = f"Results/Sweep Paras/{data_type}_{classifier_name}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, data in enumerate(valloader):
            # print(f"batch {idx+1}")
            # if idx >= batch:
            #     break
            print(len(valloader.dataset))
            original_images, original_labels = data[0].to(device), data[1].to(device)
            classified_label, original_exit = Resnet50_ee.threshold_inference_new(classifier, 0, original_images, thresholds)

            generated_images = generator(original_images)
            classified_label_gen, generated_exit = Resnet50_ee.threshold_inference_new(classifier, 0, generated_images, thresholds)

            print(f"Original Image - label: {classified_label}, \n Exit Location: {original_exit}")
            print(f"Generated Image - label: {classified_label_gen}, \n Exit Location: {generated_exit}")

            num_correct_class_original += (classified_label == category).sum().item()
            num_correct_class_generated += (classified_label_gen == category).sum().item()

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
                plt.title(f"Gen Image {i + 1}, "
                          f"label: {classified_label_gen[i]}, "
                          f"exit: {generated_exit[i]}")

                out2 = np.transpose(generated_images[i].cpu().numpy(), (1, 2, 0))
                out2_normalized = (out2 - np.min(out2)) / (np.max(out2) - np.min(out2))
                plt.imshow(out2_normalized)

                # Save the figure
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f'{save_dir}/LossPara_{EE_LOSS_PARA}_Label_{category}:{classified_label[i].item()}:{classified_label_gen[i].item()}_EE_{original_exit[i]}:{generated_exit[i]}_time_{timestamp}.png'
                plt.savefig(save_path)
                plt.close()

                plt.show()

        print(f"Number of {category}s in Original Labels: {num_correct_class_original}")
        print(f"Number of {category}s in Generated Labels: {num_correct_class_generated}")

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

def create_masked_dataset(class_number, resnet_for_cam):
    def process_and_save_data(dataloader, save_path, dataset_name):
        generated_images = []
        labels = []

        print(f"Generating images for {dataset_name} dataset...")

        for images, batch_labels in tqdm(dataloader, desc=f"Processing {dataset_name}"):
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            masked_imgs = grad_cam_mask(batch_labels, images, resnet_for_cam)
            # Move data to CPU for saving
            generated_images.append(masked_imgs.cpu())
            labels.append(batch_labels.detach().cpu())

        # Concatenate tensors to create a single dataset
        generated_images = torch.cat(generated_images, dim=0)
        labels = torch.cat(labels, dim=0)

        # Save to .pkl file
        with open(save_path, "wb") as f:
            pickle.dump({"images": generated_images, "labels": labels}, f)

        print(f"Saved {dataset_name} dataset to {save_path} with {len(generated_images)} samples")

    # get airplane/etc class index
    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_gen(root)

    train_idx, val_idx, test_idx = dataprep.get_category_index(category=class_number)  # 0 airplane, 3 cat, 8 ship
    # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
    trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=128, num_workers=2,
                                                                          train_idx=train_idx, val_idx=val_idx,
                                                                          test_idx=test_idx)
    print("Generating masked datasets...")

    process_and_save_data(trainloader, f'data_split/masked_CIFAR224_train_{class_number}.pkl', f"masked_CIFAR224_train_{class_number}")
    process_and_save_data(valloader, f'data_split/masked_CIFAR224_val_{class_number}.pkl', f"masked_CIFAR224_val_{class_number}")
    process_and_save_data(testloader, f'data_split/masked_CIFAR224_test_{class_number}.pkl', f"masked_CIFAR224_test_{class_number}")
    print("All masked datasets generated and saved successfully!")

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

    def count_exits(exits_list):
        """Count occurrences of each exit point (0-5) in the list"""
        counts = [0] * 5  # Initialize counters for exits 0-5
        for exit_point in exits_list:
            if 0 <= exit_point < 5:  # Ensure valid exit point
                counts[exit_point] += 1

        # Calculate percentages
        total = sum(counts)
        percentages = [100 * count / total if total > 0 else 0 for count in counts]

        return counts, percentages

    # Load the datasets
    print("Loading datasets...")
    test_dataset = load_data(testset_path)
    val_dataset = load_data(valset_path)
    # print(len(test_dataset))
    # print(len(val_dataset))

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)

    trainset_labels, trainset_exits = [], []
    valset_labels, valset_exits = [], []

    ori_trainset_labels, ori_trainset_exits = [], []
    ori_valset_labels, ori_valset_exits = [], []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            imgs, _ = data[0].to(device), data[1].to(device)

            # Evaluate the model on the train and validation datasets
            trainset_label, trainset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, imgs, thresholds)
            trainset_labels.append(trainset_label)
            trainset_exits.append(trainset_exit)

        for idx, data in enumerate(val_loader):
            imgs, _ = data[0].to(device), data[1].to(device)

            valset_label, valset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, imgs, thresholds)
            valset_labels.append(valset_label)
            valset_exits.append(valset_exit)

        for idx, data in enumerate(testloader):
            imgs, _ = data[0].to(device), data[1].to(device)
            ori_trainset_label, ori_trainset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, imgs, thresholds)

            ori_trainset_labels.append(ori_trainset_label)
            ori_trainset_exits.append(ori_trainset_exit)

        for idx, data in enumerate(valloader):
            imgs, _ = data[0].to(device), data[1].to(device)
            ori_valset_label, ori_valset_exit = Resnet50_ee.threshold_inference_new(classifier, 0, imgs, thresholds)

            ori_valset_labels.append(ori_valset_label)
            ori_valset_exits.append(ori_valset_exit)


    # print(f"Testset label: {trainset_labels}, \n Exit Location: {trainset_exits}")
    # print(f"Valset label: {valset_labels}, \n Exit Location: {valset_exits}")

    print(f"----Test Exit Location: {trainset_exits}")
    print(f'Original Exit Location: {ori_trainset_exits}')
    print(f"-----Val Exit Location: {valset_exits}")
    print(f'Original Exit Location: {ori_valset_exits}\n')

    # Flatten the nested lists before calculating differences
    flat_train_exits = [item for sublist in trainset_exits for item in sublist]
    flat_ori_exits = [item for sublist in ori_trainset_exits for item in sublist]

    # Count exits in both original and generated data
    gen_counts, gen_percentages = count_exits(flat_train_exits)
    ori_counts, ori_percentages = count_exits(flat_ori_exits)

    # Print exit distribution comparison
    print("\n=== Exit Distribution Analysis ===")
    print("Exit | Original Count (%) | Generated Count (%)")
    print("-" * 50)
    for i in range(5):
        print(f"Exit {i} | {ori_counts[i]} ({ori_percentages[i]:.2f}%) | {gen_counts[i]} ({gen_percentages[i]:.2f}%)")
    print("-" * 50)

    # Calculate differences
    diff = [b - a for a, b in zip(flat_train_exits, flat_ori_exits)]
    negative_count = sum(1 for value in diff if value < 0)
    positive_count = sum(1 for value in diff if value > 0)
    total_diff = len(diff)

    print(f"early or late: {diff}")
    print(f"negative count: {negative_count} of {total_diff}")
    print(f"positive count: {positive_count} of {total_diff}")
    print(f"maintain count: {total_diff - positive_count - negative_count} of {total_diff}")

def initialize_model(classifier_name, data_type):

    if classifier_name == 'B_alex':
        classifier = BranchedAlexNet()
        classifier.load_state_dict(
            torch.load(r"/home/yibo/PycharmProjects/Thesis/weights/B-Alex final/B-Alex_cifar10.pth", weights_only=True))

        classifier.to(device)
        classifier.eval()

        thresholds = [0.7, 0.8, 1.0, 0.8, 0.7]

    elif classifier_name == 'B_Vgg16':
        classifier = BranchVGG16BN()
        classifier.to(device)
        classifier = nn.DataParallel(classifier, device_ids=deviceid)
        classifier.load_state_dict(torch.load(r"weights/Vgg16bn_ee_224/Vgg16bn_epoch_15.pth", weights_only=True))
        classifier.eval()

        thresholds = [0.5, 0.5, 0.7, 0.85, 0.5]
        # thresholds = [0, 0.5, 0.7, 0.85, 0.5]
        # thresholds = [0, 0, 0.7, 0.85, 0.5]
        # thresholds = [0, 0, 0, 0.85, 0.5]

    elif classifier_name == 'B_Resnet50':
        if data_type == "CIFAR10":
            classifier = BranchedResNet50(num_classes=10).to(device)
            classifier = nn.DataParallel(classifier, device_ids=deviceid)
            classifier.load_state_dict(torch.load(r"weights/Resnet50/B-Resnet50_epoch_10.pth", weights_only=True))
            thresholds = [0.3, 0.45, 0.7, 0.05]
        elif data_type == "CIFAR100":
            classifier = BranchedResNet50(num_classes=100).to(device)
            classifier.load_state_dict(torch.load("weights/Resnet50/B-Resnet50_cifar100_best.pth", weights_only=True, map_location=device))
            thresholds = [1.0, 1.1, 1.1, 0.7]

        classifier.eval()


    # # # Load ori ResNet50 for gradcam mask
    # resnet_for_cam = ResNet_50().to(device)
    # resnet_for_cam = nn.DataParallel(resnet_for_cam, device_ids=deviceid)
    # resnet_for_cam.load_state_dict(torch.load(r"weights/Resnet50/Resnet50_ori_epoch_20.pth", weights_only=True))
    # resnet_for_cam.eval()


    return classifier, thresholds

def get_data_loaders(data_type, batch_size, num_workers, category):
    if data_type == "CIFAR10":
        # get airplane/etc class index
        root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
        dataprep = Data_prep_224_gen(root)

        train_idx, val_idx, test_idx = dataprep.get_category_index(category=category)  # 0 airplane, 3 cat, 8 ship
        trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=batch_size, num_workers=num_workers,
                                                                              train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    elif data_type == "CIFAR100":
        if super_class:
            trainloader = create_cifar100_superclass_dataloaders(data_dir='./CIFAR100', superclass_id=category,
                                                                 batch_size=batch_size, num_workers=num_workers,
                                                                 train=True, shuffle=True)

            # Create test dataloaders for all superclasses
            valloader = create_cifar100_superclass_dataloaders(data_dir='./CIFAR100', superclass_id=category,
                                                                 batch_size=batch_size, num_workers=num_workers,
                                                                 train=False, shuffle=False)
            testloader = valloader

        else:
            trainloader, valloader, testloader = load_cifar100_generator(batch_size=batch_size,
                                                                         num_workers=num_workers, class_idx=category)

    return trainloader, valloader, testloader

def refresh_generator():
    """Create a fresh generator with newly initialized weights"""
    global generator, optimizer  # Access the global variables

    # Clean up old generator memory
    if 'generator' in globals():
        generator.cpu()  # Move model to CPU first
        del generator  # Delete the model
        torch.cuda.empty_cache()  # Clear CUDA cache

    # Create new generator
    generator = Generator().to(device)
    optimizer = optim.AdamW(generator.parameters(), lr=5e-4, weight_decay=1e-5)
    return generator, optimizer


if __name__ == "__main__":

    # B_alex Alex_ee_inference
    # B_Vgg16 Vgg16_ee_inference
    # B_Resnet50 Resnet50_ee

    pl.seed_everything(2020)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    deviceid = [1]

    data_type = "CIFAR100"  # CIFAR10, CIFAR100
    super_class = False
    classifier_name = "B_Resnet50"
    batch_size = 128
    num_workers = 4

    TRAIN = True
    WHITE_BOARD_TEST = False   # actually its black board test
    categories = [4]  # [i for i in range(100)]
    EE_LOSS_PARAS = [50, 100, 300]

    classifier, thresholds = initialize_model(classifier_name, data_type)

    if TRAIN:
        for category in categories:
            for EE_LOSS_PARA in EE_LOSS_PARAS:
                generator, optimizer = refresh_generator()
                trainloader, valloader, testloader = get_data_loaders(data_type, batch_size, num_workers, category)
                train(100)
    else:
        checkpoint = torch.load('training_weights/Generator_CNN/Generator_epoch_50.pth')
        generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        category = 0
        trainloader, valloader, testloader = get_data_loaders(data_type, batch_size, num_workers, category)
        test(10, 2)  # test #batch, each batch show #num images
        gen_dataset(generator, trainloader, valloader, testloader)
        # display_gen_dataset('data_split/masked_CIFAR224_test.pkl', 'data_split/masked_CIFAR224_val.pkl', 10)
        test_gen_dataset('data_split/generated_CIFAR224_test.pkl', 'data_split/generated_CIFAR224_val.pkl')



# CIFAR100_SUPERCLASS_MAPPING = {
#     0: [4, 30, 55, 72, 95],  # aquatic mammals
#     1: [1, 32, 67, 73, 91],  # fish
#     2: [54, 62, 70, 82, 92],  # flowers
#     3: [9, 10, 16, 28, 61],  # food containers
#     4: [0, 51, 53, 57, 83],  # fruit and vegetables
#     5: [22, 39, 40, 86, 87],  # household electrical devices
#     6: [5, 20, 25, 84, 94],  # household furniture
#     7: [6, 7, 14, 18, 24],  # insects
#     8: [3, 42, 43, 88, 97],  # large carnivores
#     9: [12, 17, 37, 68, 76],  # large man-made outdoor things
#     10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
#     11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
#     12: [34, 63, 64, 66, 75],  # medium-sized mammals
#     13: [26, 45, 77, 79, 99],  # non-insect invertebrates
#     14: [2, 11, 35, 46, 98],  # people
#     15: [27, 29, 44, 78, 93],  # reptiles
#     16: [36, 50, 65, 74, 80],  # small mammals
#     17: [47, 52, 56, 59, 96],  # trees
#     18: [8, 13, 48, 58, 90],  # vehicles 1
#     19: [41, 69, 81, 85, 89]  # vehicles 2
# }


# 1     
#   optimizer = optim.AdamW(generator.parameters(), lr=5e-4, weight_decay=1e-5)
#     batch_size = 128
#     num_workers = 4
#       100, 300

