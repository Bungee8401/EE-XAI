import torch
import torch.nn as nn
import pytorch_lightning
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from CustomDataset import Data_prep_224_normal_N, load_cifar100
import time
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

class BranchVGG16BN(nn.Module):
    def __init__(self, num_classes=10):
        super(BranchVGG16BN, self).__init__()

        self.num_classes = num_classes

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # Branches for early exits
        self.branch1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 112 * 112, self.num_classes)
        )

        self.branch2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, self.num_classes)
        )

        self.branch3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, self.num_classes)
        )

        self.branch4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14,  self.num_classes)
        )

        self.branch5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, self.num_classes)
        )

    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        out_branch1 = self.branch1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        out_branch2 = self.branch2(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        out_branch3 = self.branch3(x)

        # Block 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        out_branch4 = self.branch4(x)

        # Block 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)
        out_branch5 = self.branch5(x)

        # Fully connected layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out_main = self.classifier(x)

        return out_main, out_branch1, out_branch2, out_branch3, out_branch4, out_branch5

    def extract_features(self, x):
        # Feature extraction for downstream tasks
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def initialize_model(dataset_type, batch_size, num_workers, class_idx=None):
    pytorch_lightning.seed_everything(2024)

    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create VGG16BN model with early exits with appropriate number of classes
    if dataset_type == 'cifar100':
        model = BranchVGG16BN(num_classes=100)
    else:  # cifar10
        model = BranchVGG16BN(num_classes=10)

    # Load pretrained VGG16BN model
    pretrained_vgg16bn = models.vgg16_bn(weights=True)

    # Copy weights from pretrained model to custom model
    model.conv1_1.load_state_dict(pretrained_vgg16bn.features[0].state_dict())  # conv1_1
    model.bn1_1.load_state_dict(pretrained_vgg16bn.features[1].state_dict())  # bn1_1
    model.conv1_2.load_state_dict(pretrained_vgg16bn.features[3].state_dict())  # conv1_2
    model.bn1_2.load_state_dict(pretrained_vgg16bn.features[4].state_dict())  # bn1_2

    model.conv2_1.load_state_dict(pretrained_vgg16bn.features[7].state_dict())  # conv2_1
    model.bn2_1.load_state_dict(pretrained_vgg16bn.features[8].state_dict())  # bn2_1
    model.conv2_2.load_state_dict(pretrained_vgg16bn.features[10].state_dict())  # conv2_2
    model.bn2_2.load_state_dict(pretrained_vgg16bn.features[11].state_dict())  # bn2_2

    model.conv3_1.load_state_dict(pretrained_vgg16bn.features[14].state_dict())  # conv3_1
    model.bn3_1.load_state_dict(pretrained_vgg16bn.features[15].state_dict())  # bn3_1
    model.conv3_2.load_state_dict(pretrained_vgg16bn.features[17].state_dict())  # conv3_2
    model.bn3_2.load_state_dict(pretrained_vgg16bn.features[18].state_dict())  # bn3_2
    model.conv3_3.load_state_dict(pretrained_vgg16bn.features[20].state_dict())  # conv3_3
    model.bn3_3.load_state_dict(pretrained_vgg16bn.features[21].state_dict())  # bn3_3

    model.conv4_1.load_state_dict(pretrained_vgg16bn.features[24].state_dict())  # conv4_1
    model.bn4_1.load_state_dict(pretrained_vgg16bn.features[25].state_dict())  # bn4_1
    model.conv4_2.load_state_dict(pretrained_vgg16bn.features[27].state_dict())  # conv4_2
    model.bn4_2.load_state_dict(pretrained_vgg16bn.features[28].state_dict())  # bn4_2
    model.conv4_3.load_state_dict(pretrained_vgg16bn.features[30].state_dict())  # conv4_3
    model.bn4_3.load_state_dict(pretrained_vgg16bn.features[31].state_dict())  # bn4_3

    model.conv5_1.load_state_dict(pretrained_vgg16bn.features[34].state_dict())  # conv5_1
    model.bn5_1.load_state_dict(pretrained_vgg16bn.features[35].state_dict())  # bn5_1
    model.conv5_2.load_state_dict(pretrained_vgg16bn.features[37].state_dict())  # conv5_2
    model.bn5_2.load_state_dict(pretrained_vgg16bn.features[38].state_dict())  # bn5_2
    model.conv5_3.load_state_dict(pretrained_vgg16bn.features[40].state_dict())  # conv5_3
    model.bn5_3.load_state_dict(pretrained_vgg16bn.features[41].state_dict())  # bn5_3

    # Fully connected layers
    model.classifier[0].load_state_dict(pretrained_vgg16bn.classifier[0].state_dict())  # fc1
    model.classifier[3].load_state_dict(pretrained_vgg16bn.classifier[3].state_dict())  # fc2
    # Don't load weights for the final FC layer as it has a different output size

    model = model.to(device)
    # model = nn.DataParallel(model)

    # Load the appropriate dataset
    if dataset_type == 'cifar100':
        trainloader, valloader, testloader = load_cifar100(batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         class_idx=class_idx)
    else:  # cifar10
        root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
        dataprep = Data_prep_224_normal_N(root)
        trainloader, valloader, testloader = dataprep.create_loaders(batch_size=batch_size,
                                                                   num_workers=num_workers)

    return model, device, trainloader, valloader, testloader

def train(num_epochs, dataset_type, class_idx=None):
    # Lists to store train and validation losses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_exit1 = []
    val_exit2 = []
    val_exit3 = []
    val_exit4 = []
    val_exit5 = []

    # For early stopping and best model tracking
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    early_stopping_patience = 10  # Number of epochs to wait before stopping

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time for the epoch
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(inputs)

            # Calculate individual losses
            loss_exit1 = criterion(exit1_out, labels)
            loss_exit2 = criterion(exit2_out, labels)
            loss_exit3 = criterion(exit3_out, labels)
            loss_exit4 = criterion(exit4_out, labels)
            loss_exit5 = criterion(exit5_out, labels)
            loss_main = criterion(main_out, labels)

            # Combine losses with weighted sum
            loss_all = ((1/6)*loss_exit1
                        + (1/5)*loss_exit2
                        + (1/4)*loss_exit3
                        + (1/3)*loss_exit4
                        + (1/2)*loss_exit5
                        + loss_main)            #to alleviate gradient imbalance issue; paper EECE

            loss_all.backward()
            optimizer.step()

            running_loss += loss_all.item()

            # Calculate training accuracy for this batch
            _, predicted = torch.max(main_out.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        train_accuracies.append(100 * correct_train / total_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        correct_1 = 0
        correct_2 = 0
        correct_3 = 0
        correct_4 = 0
        correct_5 = 0
        total_val = 0

        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

                # Calculate individual losses
                loss_exit1 = criterion(exit1_out, labels)
                loss_exit2 = criterion(exit2_out, labels)
                loss_exit3 = criterion(exit3_out, labels)
                loss_exit4 = criterion(exit4_out, labels)
                loss_exit5 = criterion(exit5_out, labels)
                loss_main = criterion(main_out, labels)

                # Combine losses with weighted sum
                loss_all = ((1 / 6) * loss_exit1
                            + (1 / 5) * loss_exit2
                            + (1 / 4) * loss_exit3
                            + (1 / 3) * loss_exit4
                            + (1 / 2) * loss_exit5
                            + loss_main)

                val_loss += loss_all.item()

                _, predicted_main = torch.max(main_out.data, 1)
                _, predicted_exit1 = torch.max(exit1_out.data, 1)
                _, predicted_exit2 = torch.max(exit2_out.data, 1)
                _, predicted_exit3 = torch.max(exit3_out.data, 1)
                _, predicted_exit4 = torch.max(exit4_out.data, 1)
                _, predicted_exit5 = torch.max(exit5_out.data, 1)

                total_val += labels.size(0)

                correct_val += (predicted_main == labels).sum().item()
                correct_1 += (predicted_exit1 == labels).sum().item()
                correct_2 += (predicted_exit2 == labels).sum().item()
                correct_3 += (predicted_exit3 == labels).sum().item()
                correct_4 += (predicted_exit4 == labels).sum().item()
                correct_5 += (predicted_exit5 == labels).sum().item()

        scheduler.step()

        val_loss /= len(valloader)
        val_losses.append(val_loss)

        current_val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(current_val_accuracy)
        val_exit1.append(100 * correct_1 / total_val)
        val_exit2.append(100 * correct_2 / total_val)
        val_exit3.append(100 * correct_3 / total_val)
        val_exit4.append(100 * correct_4 / total_val)
        val_exit5.append(100 * correct_5 / total_val)

        # Check if this is the best validation accuracy
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            # Save the best model so far
            save_path = f'/home/yibo/PycharmProjects/Thesis/training_weights/Vgg16bn_ee/'
            if dataset_type == 'cifar100':
                if class_idx is not None:
                    best_model_path = save_path + f'Vgg16bn_ee_cifar100_class{class_idx}_best.pth'
                else:
                    best_model_path = save_path + 'Vgg16bn_ee_cifar100_best.pth'
            else:
                best_model_path = save_path + 'Vgg16bn_ee_cifar10_best.pth'

            torch.save(best_model_state, best_model_path)
            print(f"New best validation accuracy: {best_val_accuracy:.2f}%")
        else:
            epochs_without_improvement += 1

        epoch_end_time = time.time()  # End time for the epoch
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        epoch_hours, epoch_remainder = divmod(epoch_elapsed_time, 3600)
        epoch_minutes, epoch_seconds = divmod(epoch_remainder, 60)

        print(f"Epoch [{epoch + 1}/{num_epochs}], avg Train Loss: {epoch_loss:.4f}, avg Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], main Train Accuracy: {train_accuracies[-1]:.2f}%, main Val Accuracy: {val_accuracies[-1]:.2f}%")
        print(f"Exit 1 val Accuracy: {val_exit1[-1]:.2f}%, Exit 2 val Accuracy: {val_exit2[-1]:.2f}%, Exit 3 val Accuracy: {val_exit3[-1]:.2f}%, Exit 4 val Accuracy: {val_exit4[-1]:.2f}%, Exit 5 val Accuracy: {val_exit5[-1]:.2f}%")
        print(f"Time per epoch: {int(epoch_hours)}h {int(epoch_minutes)}m {int(epoch_seconds)}s")
        print(f"Best validation accuracy so far: {best_val_accuracy:.2f}%, Epochs without improvement: {epochs_without_improvement}")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"No improvement for {early_stopping_patience} epochs. Stopping training.")
            break

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = f'/home/yibo/PycharmProjects/Thesis/training_weights/Vgg16bn_ee/'
            if dataset_type == 'cifar100':
                if class_idx is not None:
                    epoch_save_path = save_path + f'Vgg16bn_ee_cifar100_class{class_idx}_epoch_{epoch + 1}.pth'
                else:
                    epoch_save_path = save_path + f'Vgg16bn_ee_cifar100_epoch_{epoch + 1}.pth'
            else:
                epoch_save_path = save_path + f'Vgg16bn_ee_cifar10_epoch_{epoch + 1}.pth'

            torch.save(model.state_dict(), epoch_save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

    # Load the best model before saving final results
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model based on validation accuracy.")

    # Save the final trained model (best model)
    save_path = f'/home/yibo/PycharmProjects/Thesis/training_weights/Vgg16bn_ee/'
    if dataset_type == 'cifar100':
        if class_idx is not None:
            save_path += f'Vgg16bn_ee_cifar100_class{class_idx}.pth'
        else:
            save_path += 'Vgg16bn_ee_cifar100.pth'
    else:
        save_path += 'Vgg16bn_ee_cifar10.pth'

    torch.save(model.state_dict(), save_path)

    # Plotting the losses
    epoch_count = len(train_losses)
    epoch = range(1, epoch_count + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_losses, label='Train Loss')
    plt.plot(epoch, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch, train_accuracies, label='Train Accuracy')
    plt.plot(epoch, val_accuracies, label='Validation Accuracy')
    plt.plot(epoch, val_exit1, label='Exit 1 Accuracy')
    plt.plot(epoch, val_exit2, label='Exit 2 Accuracy')
    plt.plot(epoch, val_exit3, label='Exit 3 Accuracy')
    plt.plot(epoch, val_exit4, label='Exit 4 Accuracy')
    plt.plot(epoch, val_exit5, label='Exit 5 Accuracy')
    plt.axhline(y=best_val_accuracy, color='r', linestyle='--', label=f'Best Val Acc: {best_val_accuracy:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plot
    plt.savefig(f'/home/yibo/PycharmProjects/Thesis/Results/vgg16bn_ee_training_{dataset_type}.png')

    # play_sound()
    plt.show()
    print('Finished Training')
    print(f'Best validation accuracy: {best_val_accuracy:.2f}%')

def test():
    # Test the model after training
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    correct_5 = 0

    with torch.no_grad():
        for data in tqdm(testloader, desc="Testing"):
            images, labels = data[0].to(device), data[1].to(device)
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate individual losses
            loss_exit1 = criterion(exit1_out, labels)
            loss_exit2 = criterion(exit2_out, labels)
            loss_exit3 = criterion(exit3_out, labels)
            loss_exit4 = criterion(exit4_out, labels)
            loss_exit5 = criterion(exit5_out, labels)
            loss_main = criterion(main_out, labels)

            # Combine losses (average them for now)
            loss_all = ((1 / 6) * loss_exit1
                        + (1 / 5) * loss_exit2
                        + (1 / 4) * loss_exit3
                        + (1 / 3) * loss_exit4
                        + (1 / 2) * loss_exit5
                        + loss_main)  # to alleviate gradient imbalance issue; paper EECE

            test_loss += loss_all.item()

            _, predicted_main = torch.max(main_out.data, 1)
            _, predicted_exit1 = torch.max(exit1_out.data, 1)
            _, predicted_exit2 = torch.max(exit2_out.data, 1)
            _, predicted_exit3 = torch.max(exit3_out.data, 1)
            _, predicted_exit4 = torch.max(exit4_out.data, 1)
            _, predicted_exit5 = torch.max(exit5_out.data, 1)

            total_test += labels.size(0)
            correct_test += (predicted_main == labels).sum().item()
            correct_1 += (predicted_exit1 == labels).sum().item()
            correct_2 += (predicted_exit2 == labels).sum().item()
            correct_3 += (predicted_exit3 == labels).sum().item()
            correct_4 += (predicted_exit4 == labels).sum().item()
            correct_5 += (predicted_exit5 == labels).sum().item()


    test_loss /= len(testloader)
    test_accuracy = 100 * correct_test / total_test
    exit1_accuracy = 100 * correct_1 / total_test
    exit2_accuracy = 100 * correct_2 / total_test
    exit3_accuracy = 100 * correct_3 / total_test
    exit4_accuracy = 100 * correct_4 / total_test
    exit5_accuracy = 100 * correct_5 / total_test

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
          f"Exit 1 Accuracy: {exit1_accuracy:.2f}%, Exit 2 Accuracy: {exit2_accuracy:.2f}%, "
          f"Exit 3 Accuracy: {exit3_accuracy:.2f}%, Exit 4 Accuracy: {exit4_accuracy:.2f}%, "
          f"Exit 5 Accuracy: {exit5_accuracy:.2f}%")

def simple_inference(model, dataloader):
    model.eval()
    correct_main = 0
    correct_exit1 = 0
    correct_exit2 = 0
    correct_exit3 = 0
    correct_exit4 = 0
    correct_exit5 = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate predictions for each exit
            _, predicted_main = torch.max(main_out.data, 1)
            _, predicted_exit1 = torch.max(exit1_out.data, 1)
            _, predicted_exit2 = torch.max(exit2_out.data, 1)
            _, predicted_exit3 = torch.max(exit3_out.data, 1)
            _, predicted_exit4 = torch.max(exit4_out.data, 1)
            _, predicted_exit5 = torch.max(exit5_out.data, 1)

            correct_main += (predicted_main == labels).sum().item()
            correct_exit1 += (predicted_exit1 == labels).sum().item()
            correct_exit2 += (predicted_exit2 == labels).sum().item()
            correct_exit3 += (predicted_exit3 == labels).sum().item()
            correct_exit4 += (predicted_exit4 == labels).sum().item()
            correct_exit5 += (predicted_exit5 == labels).sum().item()

    accuracy_main = 100 * correct_main / total
    accuracy_exit1 = 100 * correct_exit1 / total
    accuracy_exit2 = 100 * correct_exit2 / total
    accuracy_exit3 = 100 * correct_exit3 / total
    accuracy_exit4 = 100 * correct_exit4 / total
    accuracy_exit5 = 100 * correct_exit5 / total

    return [accuracy_exit1, accuracy_exit2, accuracy_exit3, accuracy_exit4, accuracy_exit5, accuracy_main]

def threshold_finder(model, dataloader, initial_thresholds, accuracy, step, tolerance):
    optimal_thresholds = initial_thresholds.copy()
    best_accuracies = [0] * len(initial_thresholds)
    best_exit_ratios = [0] * len(initial_thresholds)
    goal_accuracy = [acc - tolerance for acc in accuracy]
    upper_acc = accuracy[-1]

    for i in range(len(initial_thresholds)):
        print(f"----finding optimal threshold for early exit point----{i+1}")
        while accuracy[i] < upper_acc and 0 < optimal_thresholds[i] <= 2.3:
            model.eval()
            accuracy, exit_ratios = threshold_inference(model, dataloader, optimal_thresholds)

            best_accuracies[i] = accuracy[i]
            best_exit_ratios[i] = exit_ratios[i]
            optimal_thresholds[i] -= step
            print(f"Current thresh: {optimal_thresholds[i]}, Accuracy: {best_accuracies[i]}, Exit Ratio: {best_exit_ratios[i]}")

        # Revert the last step
        if optimal_thresholds[i] == 2.2:
            optimal_thresholds[i] += 0.05
        else:
            optimal_thresholds[i] += step

        print(f"Optimal thresh: {optimal_thresholds[i]}, Accuracy: {best_accuracies[i]}, Exit Ratio: {best_exit_ratios[i]}")

        optimal_thresholds[i] = max(0.01, min(2.2, optimal_thresholds[i]))

    return optimal_thresholds, best_accuracies, best_exit_ratios

def threshold_inference(model, dataloader, exit_thresholds):
    model.eval()

    correct_main = 0
    correct_exit1 = 0
    correct_exit2 = 0
    correct_exit3 = 0
    correct_exit4 = 0
    correct_exit5 = 0

    total = 0
    exit_counts = [0, 0, 0, 0, 0, 0]  # Six exits including main

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate softmax and entropy for each exit
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

            # Determine exit points based on thresholds
            for i in range(labels.size(0)):
                if entropy_exit1[i] < exit_thresholds[0]:
                    _, predicted = torch.max(exit1_out[i].data, 0)
                    exit_counts[0] += 1
                    if predicted == labels[i]:
                        correct_exit1 += 1

                elif entropy_exit2[i] < exit_thresholds[1]:
                    _, predicted = torch.max(exit2_out[i].data, 0)
                    exit_counts[1] += 1
                    if predicted == labels[i]:
                        correct_exit2 += 1

                elif entropy_exit3[i] < exit_thresholds[2]:
                    _, predicted = torch.max(exit3_out[i].data, 0)
                    exit_counts[2] += 1
                    if predicted == labels[i]:
                        correct_exit3 += 1

                elif entropy_exit4[i] < exit_thresholds[3]:
                    _, predicted = torch.max(exit4_out[i].data, 0)
                    exit_counts[3] += 1
                    if predicted == labels[i]:
                        correct_exit4 += 1

                elif entropy_exit5[i] < exit_thresholds[4]:
                    _, predicted = torch.max(exit5_out[i].data, 0)
                    exit_counts[4] += 1
                    if predicted == labels[i]:
                        correct_exit5 += 1

                else:
                    _, predicted = torch.max(main_out[i].data, 0)
                    exit_counts[5] += 1
                    if predicted == labels[i]:
                        correct_main += 1

    # Calculate accuracy and exit ratios
    exit1_accuracy = 100 * correct_exit1 / exit_counts[0] if exit_counts[0] > 0 else 0
    exit2_accuracy = 100 * correct_exit2 / exit_counts[1] if exit_counts[1] > 0 else 0
    exit3_accuracy = 100 * correct_exit3 / exit_counts[2] if exit_counts[2] > 0 else 0
    exit4_accuracy = 100 * correct_exit4 / exit_counts[3] if exit_counts[3] > 0 else 0
    exit5_accuracy = 100 * correct_exit5 / exit_counts[4] if exit_counts[4] > 0 else 0
    main_accuracy = 100 * correct_main / exit_counts[5] if exit_counts[5] > 0 else 0

    exit_ratios = [100 * count / total for count in exit_counts]

    return [exit1_accuracy, exit2_accuracy, exit3_accuracy, exit4_accuracy, exit5_accuracy, main_accuracy], exit_ratios

def threshold_inference_single_image(model, image, exit_thresholds):
    """Process a single image and determine which exit to use based on thresholds"""
    model.eval()

    with torch.no_grad():
        main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(image)

        # Calculate softmax and entropy for each exit
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

        # Determine exit point based on entropy thresholds
        if entropy_exit1[0] < exit_thresholds[0]:
            _, predicted = torch.max(exit1_out.data, 1)
            exit_point = "Exit 1"
        elif entropy_exit2[0] < exit_thresholds[1]:
            _, predicted = torch.max(exit2_out.data, 1)
            exit_point = "Exit 2"
        elif entropy_exit3[0] < exit_thresholds[2]:
            _, predicted = torch.max(exit3_out.data, 1)
            exit_point = "Exit 3"
        elif entropy_exit4[0] < exit_thresholds[3]:
            _, predicted = torch.max(exit4_out.data, 1)
            exit_point = "Exit 4"
        elif entropy_exit5[0] < exit_thresholds[4]:
            _, predicted = torch.max(exit5_out.data, 1)
            exit_point = "Exit 5"
        else:
            _, predicted = torch.max(main_out.data, 1)
            exit_point = "Main Exit"

    return predicted.item(), exit_point

def ee_inference():
    initial_thresholds = [1.0] * 5  # Start with a reasonable threshold

    initial_accuracy = simple_inference(model, testloader)
    print(f"Simple inference accuracy: {initial_accuracy}")

    # Dis/enable threshold finder
    FIND_THRESHOLD = True
    if FIND_THRESHOLD:
        optimal_thresholds, best_accuracies, best_exit_ratios = threshold_finder(model, trainloader, initial_thresholds,
                                                                             initial_accuracy, step=0.1,
                                                                             tolerance=10)
        print("---------------------------------")
        print(f"Threshold finder optimal thresholds: {optimal_thresholds}")
        print(f"Threshold finder best accuracies: {best_accuracies}")
        print(f"Threshold finder exit ratios: {best_exit_ratios}")
    else:
        optimal_thresholds = [0.3, 0.45, 0.6, 0.75, 0.9]  # Example thresholds
        print(f"Using predefined thresholds: {optimal_thresholds}")

    # Do threshold inference
    accuracy, exit_ratios = threshold_inference(model, testloader, optimal_thresholds)
    print("---------------------------------")
    print(f"Threshold inference accuracy: {accuracy}")
    print(f"Threshold inference exit ratios: {exit_ratios}")

def white_board_test(image_path):
    # Load and preprocess a single image for testing
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Define the same transforms used for the test dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()  # Keep a copy for display
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Optimal thresholds for the VGG16BN
    optimal_thresholds = [0.3, 0.45, 0.6, 0.75, 0.9]  # Example thresholds - replace with your actual thresholds

    # Get prediction and exit point
    predicted_class, exit_point = threshold_inference_single_image(model, image_tensor, optimal_thresholds)

    # Define class names based on dataset
    if hasattr(testloader.dataset, 'classes'):
        class_names = testloader.dataset.classes
    else:
        # Default to CIFAR-10 class names if not available
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if predicted_class < len(class_names):
        class_name = class_names[predicted_class]
    else:
        class_name = f"Class {predicted_class}"

    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title(f'VGG16BN EE Prediction\n'
              f'Prediction: {class_name}, Exit: {exit_point}')
    plt.show()

if __name__ == '__main__':
    pytorch_lightning.seed_everything(2024)

    # Select dataset type
    dataset_type = 'cifar100'  # 'cifar10' or 'cifar100'
    class_idx = None  # Set to a class index (0-99) for CIFAR-100 single class, or None for all classes

    batch_size = 128
    num_workers = 4

    model, device, trainloader, valloader, testloader = initialize_model(
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        class_idx=class_idx
    )

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # TRAIN = True
    TRAIN = False
    if TRAIN:
        train(100, dataset_type, class_idx=None)
        test()
    else:
        # Adjust path depending on dataset type
        if dataset_type == 'cifar100':
            weight_path = "weights/Vgg16bn_ee_224/Vgg16bn_ee_cifar100_best.pth"
        elif dataset_type == 'cifar10':
            weight_path = "weights/Vgg16bn_ee_224/Vgg16bn_epoch_15.pth"

        model.load_state_dict(torch.load(weight_path, weights_only=True))
        test()
        # ee_inference()  # Uncomment to run early exit inference
        # white_board_test("path/to/test/image.jpg")  # Uncomment to test with a specific image


