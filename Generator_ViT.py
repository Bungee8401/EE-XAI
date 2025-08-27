import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import datetime
import ViT_ee
from ViT_ee import create_vit_base_16_ee, TinyImageNetValDataset, download_and_prepare_tiny_imagenet, load_single_class


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

def compute_loss(model, generated, original_img, target_label):
    # 早退损失
    if classifier_name == 'ViT_ee':
        if head_type == "CNN_ignore":
            main_out, output9, output6, output3 = classifier(generated)
            exit_confidences = compute_entropy_3(output9, output6, output3)

    exit_loss = torch.mean(exit_confidences)  # the mean of first exit's confidence, min it -> encourage early exit

    if classifier_name == 'ViT_ee':
        gen_feature = classifier.extract_features(generated)
        ori_feature = classifier.extract_features(original_img)  # Get the final output for original image


    cls_loss1 = nn.MSELoss()(gen_feature, ori_feature)
    cls_loss2 = nn.CrossEntropyLoss()(main_out, target_label)
    sim_loss = nn.L1Loss()(generated, original_img)

    return EE_LOSS_PARA * exit_loss  +  500 * cls_loss2 +  500 * cls_loss1 + 500 * sim_loss

def compute_entropy_3(exit1_out, exit2_out, exit3_out):
    softmax_exit1 = F.softmax(exit1_out, dim=1)
    entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

    softmax_exit2 = F.softmax(exit2_out, dim=1)
    entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

    softmax_exit3 = F.softmax(exit3_out, dim=1)
    entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

    return torch.stack([entropy_exit1, entropy_exit2, entropy_exit3], dim=0)

def train(num_epoch):
    generator.train()
    loss_log_path = r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224/log.txt"

    # Create directory for logs if it doesn't exist
    os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)

    for epoch in range(num_epoch):
        epoch_loss = 0  # Reset loss for each epoch
        num_batches = 0  # Reset batch count for each epoch

        for idx, data in enumerate(trainloader):
            original_images, original_labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            generated_images = generator(original_images)
            loss = compute_loss(classifier, generated_images, original_images, original_labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epoch}, Training Loss: {avg_loss}")

        # Append the average loss to the local file
        with open(loss_log_path, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Epoch {epoch + 1}: {avg_loss}\n")

        if not os.path.exists(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224"):
                os.makedirs(r"/home/yibo/PycharmProjects/Thesis/training_weights/Generator224")
        if (epoch+1) % 10 == 0:
            torch.save({
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'training_weights/Generator224/Generator_epoch_{epoch+1}.pth')
            test(10,1)
            white_board_test()

    print(f"Training complete for class {category}. ")

def white_board_test():
    """
    Function to perform white board test - generates images from a blank input
    and saves the results to visualize what patterns the generator has learned.
    """
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

        # Get prediction from classifier based on model type
        if classifier_name == 'ViT_ee':
            # Use threshold_inference from the ViT_ee file
            classified_label_gen, generated_exit = ViT_ee.threshold_inference_vit(classifier, generated_images, thresholds)
            # Handle potential tensor vs list/scalar issues
            if isinstance(classified_label_gen, torch.Tensor) and classified_label_gen.numel() == 1:
                label_item = classified_label_gen.item()
            elif isinstance(classified_label_gen, list) and len(classified_label_gen) > 0:
                label_item = classified_label_gen[0]
            else:
                label_item = 0

            if isinstance(generated_exit, list) and len(generated_exit) > 0:
                exit_item = generated_exit[0]
            elif isinstance(generated_exit, torch.Tensor) and generated_exit.numel() == 1:
                exit_item = generated_exit.item()
            else:
                exit_item = 0


        torchvision.utils.save_image(generated_images[0],
                                    f'{save_dir}/LossPara_{EE_LOSS_PARA}_Label_{category}:{label_item}_EE_{exit_item}_time_{timestamp}.png',
                                    normalize=True)

        print(f"White Board Test - Generated Image - label: {classified_label_gen}, Exit Location: {generated_exit}")

        # Display the image
        # plt.figure(figsize=(6, 6))
        # plt.title(f"White Board Test Result\nLabel: {classified_label_gen[0]}, Exit: {generated_exit[0]}")
        # out = np.transpose(generated_images_crop[0].cpu().numpy(), (1, 2, 0))
        # out_normalized = (out - np.min(out)) / (np.max(out) - np.min(out))
        # plt.imshow(out_normalized)
        # plt.show()

def test(batch, num):
    generator.eval()
    classifier.eval()
    num_forced_class_original = 0
    num_forced_class_generated = 0

    save_dir = f"Results/Sweep Paras/{data_type}_{classifier_name}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, data in enumerate(testloader):
            print(f"batch {idx+1}")
            if idx >= batch:
                break

            original_images, original_labels = data[0].to(device), data[1].to(device)

            # Process original images based on classifier type
            if classifier_name == 'ViT_ee':
                classified_label, original_exit = ViT_ee.threshold_inference_vit(classifier, original_images, thresholds)

            generated_images = generator(original_images)

            # Process generated images based on classifier type
            if classifier_name == 'ViT_ee':
                classified_label_gen, generated_exit = ViT_ee.threshold_inference_vit(classifier, generated_images, thresholds)


            print(f"Original Image - label: {classified_label}, \n Exit Location: {original_exit}")
            print(f"Generated Image - label: {classified_label_gen}, \n Exit Location: {generated_exit}")

            # Safely handle comparison and counting
            if isinstance(classified_label, torch.Tensor):
                num_forced_class_original += sum(1 for label in classified_label if label.item() == category)
            elif isinstance(classified_label, list):
                num_forced_class_original += sum(1 for label in classified_label if label == category)
            else:
                num_forced_class_original += 1 if classified_label == category else 0

            if isinstance(classified_label_gen, torch.Tensor):
                num_forced_class_generated += sum(1 for label in classified_label_gen if label.item() == category)
            elif isinstance(classified_label_gen, list):
                num_forced_class_generated += sum(1 for label in classified_label_gen if label == category)
            else:
                num_forced_class_generated += 1 if classified_label_gen == category else 0

            for i in range(min(num, len(original_images))): # show x imgs
                plt.figure(figsize=(8, 4))

                # Safe access to classified_label and original_exit
                if isinstance(classified_label, torch.Tensor):
                    label_text = classified_label[i].item() if i < len(classified_label) else "N/A"
                elif isinstance(classified_label, list):
                    label_text = classified_label[i] if i < len(classified_label) else "N/A"
                else:
                    label_text = classified_label

                if isinstance(original_exit, list):
                    exit_text = original_exit[i] if i < len(original_exit) else "N/A"
                elif isinstance(original_exit, torch.Tensor):
                    exit_text = original_exit[i].item() if i < len(original_exit) else "N/A"
                else:
                    exit_text = original_exit

                plt.subplot(1, 2, 1)
                plt.title(f"Original Image {i + 1}, "
                          f"label: {label_text}, "
                          f"exit: {exit_text}")

                out1 = np.transpose(original_images[i].cpu().numpy(), (1, 2, 0))
                out1_normalized = (out1 - np.min(out1)) / (np.max(out1) - np.min(out1))

                plt.imshow(out1_normalized)

                # Safe access to classified_label_gen and generated_exit
                if isinstance(classified_label_gen, torch.Tensor):
                    gen_label_text = classified_label_gen[i].item() if i < len(classified_label_gen) else "N/A"
                elif isinstance(classified_label_gen, list):
                    gen_label_text = classified_label_gen[i] if i < len(classified_label_gen) else "N/A"
                else:
                    gen_label_text = classified_label_gen

                if isinstance(generated_exit, list):
                    gen_exit_text = generated_exit[i] if i < len(generated_exit) else "N/A"
                elif isinstance(generated_exit, torch.Tensor):
                    gen_exit_text = generated_exit[i].item() if i < len(generated_exit) else "N/A"
                else:
                    gen_exit_text = generated_exit

                plt.subplot(1, 2, 2)
                plt.title(f"Gen Image {i + 1}, "
                          f"label: {gen_label_text}, "
                          f"exit: {gen_exit_text}")

                out2 = np.transpose(generated_images[i].cpu().numpy(), (1, 2, 0))
                out2_normalized = (out2 - np.min(out2)) / (np.max(out2) - np.min(out2))
                plt.imshow(out2_normalized)

                # Save the figure
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f'{save_dir}/LossPara_{EE_LOSS_PARA}_Label_{category}:{label_text}:{gen_label_text}_EE_{exit_text}:{gen_exit_text}_time_{timestamp}.png'
                plt.savefig(save_path)
                plt.close()

                plt.show()

        print(f"Number of {category}s in Original Labels: {num_forced_class_original}")
        print(f"Number of {category}s in Generated Labels: {num_forced_class_generated}")

def initialize_model(classifier_name, data_type):

    if classifier_name == 'ViT_ee':
        thresholds = [0.99, 0.99, 0.90] # acc limit
        classifier = create_vit_base_16_ee(n_classes=200).to(device)
        weights_path = f'weights/vit_ee_{data_type}_{head_type}_best.pth'
        classifier.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        classifier.eval()


    generator = Generator().to(device)

    return generator, classifier, thresholds

def get_data_loaders(data_type, category, batch_size, num_workers):

    if data_type == "tinyimagenet":
        trainloader = load_single_class(
            root_dir='Tiny Imagenet/tiny-imagenet-200',
            class_idx=category,
            split='train',
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

        # Create DataLoader for validation set
        valloader = load_single_class(
            root_dir='Tiny Imagenet/tiny-imagenet-200',
            class_idx=category,
            split='val',
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

        testloader = valloader

    return trainloader, valloader, testloader



if __name__ == "__main__":

    pl.seed_everything(2020)

    head_type = "CNN_ignore"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_type = "tinyimagenet"
    classifier_name = "ViT_ee"
    batch_size = 64
    num_workers = 4

    EE_LOSS_PARAS = [300, 500]
    categories = [i for i in range(200)]

    TRAIN = True
    WHITE_BOARD_TEST = False

    generator, classifier, thresholds = initialize_model(classifier_name, data_type)
    optimizer = optim.Adam(generator.parameters(), lr=5e-4)  # 1e-4 is not good

    if TRAIN:
        for EE_LOSS_PARA in EE_LOSS_PARAS:
            for category in categories:
                trainloader, valloader, testloader = get_data_loaders(data_type, category, batch_size, num_workers)
                train(50)
    else:
        checkpoint = torch.load('training_weights/Generator224/Generator_epoch_50.pth')
        # checkpoint = torch.load('weights/Generator224/resnet50/9_truck/Generator_epoch_40.pth')
        generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if WHITE_BOARD_TEST:
        white_board_test()
    else:
        test(10, 2)  # test #batch, each batch show #num images
