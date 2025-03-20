import torch
import torch.nn as nn
import pytorch_lightning
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from CustomDataset import Data_prep_224_normal_N
import time
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import requests

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


def initialize_model():
    pytorch_lightning.seed_everything(2024)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create AlexNet model with early exits
    model = BranchVGG16BN(num_classes=10)

    # Load pretrained AlexNet model
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
    # model.classifier[6].load_state_dict(pretrained_vgg16bn.classifier[6].state_dict())  # fc3

    model = model.to(device)
    model = nn.DataParallel(model)

    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=32, num_workers=8)


    return model, device, trainloader, valloader, testloader

def train(num_epochs):
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
            # print(i)
            optimizer.zero_grad()

            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(inputs)

            # Calculate individual losses
            loss_exit1 = criterion(exit1_out, labels)
            loss_exit2 = criterion(exit2_out, labels)
            loss_exit3 = criterion(exit3_out, labels)
            loss_exit4 = criterion(exit4_out, labels)
            loss_exit5 = criterion(exit5_out, labels)
            loss_main = criterion(main_out, labels)

            # Combine losses (average or weighted sum)
            # loss_all = (loss_exit1 + loss_exit2 + loss_main) / 3
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

                # Combine losses (average them for now)
                # loss_all = (loss_exit1 + loss_exit2 + loss_main) / 3
                loss_all = ((1 / 6) * loss_exit1
                            + (1 / 5) * loss_exit2
                            + (1 / 4) * loss_exit3
                            + (1 / 3) * loss_exit4
                            + (1 / 2) * loss_exit5
                            + loss_main)  # to alleviate gradient imbalance issue; paper EECE

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

        val_accuracies.append(100 * correct_val / total_val)
        val_exit1.append(100 * correct_1 / total_val)
        val_exit2.append(100 * correct_2 / total_val)
        val_exit3.append(100 * correct_3 / total_val)
        val_exit4.append(100 * correct_4 / total_val)
        val_exit5.append(100 * correct_5 / total_val)

        epoch_end_time = time.time()  # End time for the epoch
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        epoch_hours, epoch_remainder = divmod(epoch_elapsed_time, 3600)
        epoch_minutes, epoch_seconds = divmod(epoch_remainder, 60)

        print(f"Epoch [{epoch + 1}/{num_epochs}], avg Train Loss: {epoch_loss:.4f}, avg Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], main Train Accuracy: {train_accuracies[-1]:.2f}%, main Val Accuracy: {val_accuracies[-1]:.2f}%")
        print(f"Exit 1 val Accuracy: {val_exit1[-1]:.2f}%, Exit 2 val Accuracy: {val_exit2[-1]:.2f}%, Exit 3 val Accuracy: {val_exit3[-1]:.2f}%, Exit 4 val Accuracy: {val_exit4[-1]:.2f}%, Exit 5 val Accuracy: {val_exit5[-1]:.2f}%")
        print(f"Time per epoch: {int(epoch_hours)}h {int(epoch_minutes)}m {int(epoch_seconds)}s")


        #save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f'/home/yibo/PycharmProjects/Thesis/training_weights/Vgg16bn_ee'
                       f'/Vgg16bn_ee_small_epoch_{epoch + 1}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    # Plotting the losses
    epoch = range(1, len(train_losses) + 1)

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
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()

    # play_sound()
    plt.show()
    print('Finished Training')

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
        for data in testloader:
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

if __name__ == '__main__':

    model, device, trainloader, valloader, testloader = initialize_model()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    TRAIN = False
    if TRAIN:
        train(20)
        test()
    else:
        model.load_state_dict(torch.load(r"weights/Vgg16bn_ee_224/Vgg16bn_epoch_15.pth", weights_only=True))
        test()
        # show_images_from_valloader(valloader)

