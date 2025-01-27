import pytorch_lightning
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import winsound
import time
from CustomDataset import Data_prep_224_normal_N

class BranchedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BranchedAlexNet, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        # Branch 1 after 1st group
        self.branch1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(46656, 1024),
            nn.Linear(1024, self.num_classes),
        )

        # Branch 2 after 2nd group
        self.branch2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32448, 1024),
            nn.Linear(1024, self.num_classes),
        )

        # Branch 3 after 3rd group
        self.branch3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384 * 13 * 13, 1024),
            nn.Linear(1024, self.num_classes),
        )

        # Branch 4 after 4th group
        self.branch4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 13 * 13, 1024),
            nn.Linear(1024, self.num_classes),
        )

        # Branch 5 after 5th group
        self.branch5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        out_branch1 = self.branch1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        out_branch2 = self.branch2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        out_branch3 = self.branch3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        out_branch4 = self.branch4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        out_branch5 = self.branch5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out_main = self.classifier(x)

        return out_main, out_branch1, out_branch2, out_branch3, out_branch4, out_branch5

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def initialize_model():
    pytorch_lightning.seed_everything(2024)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create AlexNet model with early exits
    model = BranchedAlexNet(num_classes=10)

    # Load pretrained AlexNet model
    pretrained_alexnet = models.alexnet(weights=True)

    # Copy the weights from the pretrained AlexNet to your model
    model.conv1.load_state_dict(pretrained_alexnet.features[0].state_dict())
    model.conv2.load_state_dict(pretrained_alexnet.features[3].state_dict())
    model.conv3.load_state_dict(pretrained_alexnet.features[6].state_dict())
    model.conv4.load_state_dict(pretrained_alexnet.features[8].state_dict())
    model.conv5.load_state_dict(pretrained_alexnet.features[10].state_dict())

    model = model.to(device)


    # Define transforms for data augmentation and normalization
    # transform_train = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize to 224x224
    #     transforms.RandomCrop(224, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(degrees=15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize to 224x224
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # full_trainset = torchvision.datasets.CIFAR10(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', train=True,
    #                                              download=True, transform=transform_train)
    # trainset, valset = torch.utils.data.random_split(full_trainset, [40000, 10000])
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', train=False,
    #                                        download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    root = 'D:/Study/Module/Master Thesis/dataset/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=100, num_workers=2)

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


        #save every 10 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f'D:/Study/Module/Master Thesis/trained_models/B-Alex_cifar10_epoch_{epoch + 1}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    # Save the trained model
    torch.save(model.state_dict(), 'D:/Study/Module/Master Thesis/trained_models/B-Alex_cifar10.pth')

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

    play_sound()
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

def play_sound():
    for _ in range(10):
        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)

def show_images_from_valloader(valloader, num_images=10):
    import matplotlib.pyplot as plt
    import numpy as np

    # Get a batch of validation images
    for images, labels in valloader:
        break

    # Show images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.axis('off')
        ax.set_title(f'Label: {labels[idx].item()}')

    plt.show()


if __name__ == '__main__':

    model, device, trainloader, valloader, testloader = initialize_model()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    TRAIN = False
    if TRAIN:
        train(50)
        test()
    else:
        model.load_state_dict(torch.load(r"D:\Study\Module\Master Thesis\trained_models\B-Alex final\B-Alex_cifar10.pth", weights_only=True))
        test()
        show_images_from_valloader(valloader)