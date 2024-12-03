import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16_bn
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support
import json

class VGG16BNWithEarlyExit(nn.Module):
    def __init__(self, num_classes=10, exit_threshold=0.9):
        super(VGG16BNWithEarlyExit, self).__init__()
        self.vgg16_bn = vgg16_bn(weights= None)
        self.vgg16_bn.classifier[6] = nn.Linear(4096, num_classes)
        self.exit_threshold = exit_threshold

        # Define intermediate classifiers
        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.vgg16_bn.features[:16](x)
        exit1_out = self.exit1(x)

        x = self.vgg16_bn.features[16:23](x)
        exit2_out = self.exit2(x)


        x = self.vgg16_bn.features[23:](x)
        x = self.vgg16_bn.avgpool(x)
        x = torch.flatten(x, 1)
        main_out = self.vgg16_bn.classifier(x)


        return exit1_out, exit2_out, main_out


def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='D:\Study\Module\Master Thesis\dataset\CIFAR10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='D:\Study\Module\Master Thesis\dataset\CIFAR10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize the VGG16 model with early exit
    model = VGG16BNWithEarlyExit(num_classes=10, exit_threshold=0.9)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # Training the model
    num_epochs = 200
    train_losses = []
    val_losses = []
    acc = []

    for epoch in range(num_epochs):
        model.train()
        print(f'-----{epoch + 1}--training------')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            exit1_out, exit2_out, main_out = model(inputs)

            # Calculate individual losses
            loss_exit1 = criterion(exit1_out, labels)
            loss_exit2 = criterion(exit2_out, labels)
            loss_main = criterion(main_out, labels)

            # Combine losses (average them for now, cuz in BrachyNet paper,V-A p.5, only 1% acc increase if weighted differently)
            loss_all = (loss_exit1 + loss_exit2 + loss_main) / 3

            loss_all.backward()
            optimizer.step()

            running_loss += loss_all.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Evaluate the model after each epoch
        model.eval()
        print(f'-----{epoch + 1}--evaluating------')
        correct_1 = 0
        correct_2 = 0
        correct_main = 0
        total = 0
        # exit1_count = 0
        # exit2_count = 0
        # main_count = 0
        # exit_point = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                exit1_out, exit2_out, main_out = model(images)

                # Calculate individual losses
                loss_exit1 = criterion(exit1_out, labels)
                loss_exit2 = criterion(exit2_out, labels)
                loss_main = criterion(main_out, labels)

                # Combine losses (average them for now)
                loss_all = (loss_exit1 + loss_exit2 + loss_main) / 3
                val_running_loss += loss_all.item() * images.size(0)

                # acc of each branch
                _, predicted_1 = torch.max(exit1_out.data, 1)
                _, predicted_2 = torch.max(exit2_out.data, 1)
                _, predicted_main = torch.max(main_out.data, 1)
                total += labels.size(0)
                correct_1 += (predicted_1 == labels).sum().item()
                correct_2 += (predicted_2 == labels).sum().item()
                correct_main += (predicted_main == labels).sum().item()

        val_loss = val_running_loss / len(testloader.dataset)
        val_losses.append(val_loss)
        print(f'Validation Loss after epoch {epoch + 1}: {val_loss:.3f}')

        accuracy_main = 100 * correct_main / total
        accuracy_1 = 100 * correct_1 / total
        accuracy_2 = 100 * correct_2 / total
        acc.append(accuracy_main)

        # exit1_ratio = 100 * exit1_count / total
        # exit2_ratio = 100 * exit2_count / total
        # main_ratio = 100 * main_count / total


        print(f'Acc_1 after epoch {epoch + 1}: {accuracy_1:.2f}% {correct_1} / {total}')
        print(f'Acc_2 after epoch {epoch + 1}: {accuracy_2:.2f}% {correct_2} / {total}')
        print(f'Acc_main after epoch {epoch + 1}: {accuracy_main:.2f}% {correct_main} / {total}')

        # print(f'Exit 1 Ratio: {exit1_ratio:.2f}% {exit1_count} / {total}')
        # print(f'Exit 2 Ratio: {exit2_ratio:.2f}% {exit2_count} / {total}')
        # print(f'Main Exit Ratio: {main_ratio:.2f}% {main_count} / {total}')

        # Step the scheduler
        scheduler.step()
    print('----------------------------------------------------------------')

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'D:/Study/Module/Master Thesis/trained_models/vgg16_bn_branches_cifar10.pth')

    # Save training and validation losses
    with open('D:/Study/Module/Master Thesis/trained_models/vgg6bn_branches_loss.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f)

    # Plot training and validation loss


    # plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), acc, label='Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()


#todo: exit ratio somehow doesnt work, fix it (line 136-142)

#todo: total number is not 100% correct. it only shows how many samples can be classified correctly (line164-170

#todo: fast inference algorithm in another pyton file

    # exit1_confidence = entropy(exit1_softmax)
    # exit2_softmax = nn.functional.softmax(exit2_out, dim=1)
    # exit2_confidence = entropy(exit2_softmax)
    # if exit1_confidence.max() > self.exit_threshold:
    #     return exit1_out, None, None
    #
    # if exit2_confidence.max() > self.exit_threshold:
    #     return exit1_out, exit2_out, None
    #
    # return exit1_out, exit2_out, main_out
    # make sure all exit_points are reached and have a value

    # def entropy(tensor):
    #     # Ensure the tensor is a probability distribution along the rows (normalize if necessary)
    #     tensor = tensor / tensor.sum(dim=1, keepdim=True)
    #
    #     # Avoid log(0) by adding a small value epsilon
    #     # epsilon = 1e-10
    #     # tensor = tensor + epsilon
    #
    #     # Calculate entropy for each row
    #     entropy = -torch.sum(tensor * torch.log2(tensor), dim=1, keepdim=True)
    #     return entropy

#todo: Count exit points early_exit, main_exit
    # if exit_point == 1:
    #     exit1_count += 1
    # elif exit_point == 2:
    #     exit2_count += 1
    # else:
    #     main_count += 1