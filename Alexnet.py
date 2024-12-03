import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(2024)  #for reproducibility
    if device.type == 'cuda':
        torch.cuda.manual_seed(2024)
        torch.cuda.manual_seed_all(2024)

        # Define transforms for data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='D:\Study\Module\Master Thesis\dataset\CIFAR10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='D:\Study\Module\Master Thesis\dataset\CIFAR10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Create VGG16 model with pretrained=False to initialize without weights
    model = alexnet(weights=None)
    # Modify the final fully connected layer to match CIFAR-10's 10 classes
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
    # model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Modified first layer

    model.load_state_dict(torch.load(r"D:\Study\Module\Master Thesis\trained_models\Alexnet_cifar10_90%.pth"))

    # Move the model to the device (GPU or CPU)
    model = model.to(device)
    # print(model)
    # Lists to store train and validation losses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    total_train = 0
    correct_train = 0
    num_epochs = 10
    learning_rate = 0.0005

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # optimizer = optim.AdamW([
    #     {'params': model.features.parameters(), 'lr': 1e-4},  # Lower rate for features
    #     {'params': model.classifier.parameters(), 'lr': 1e-3}  # Higher rate for classifier
    # ], lr=1e-3, weight_decay=5e-4)

    # Define the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0  # Reset for each epoch
        total_train = 0  # Reset for each epoch

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy for this batch
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        scheduler.step()

        val_loss /= len(testloader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

        #save every 10 epochs
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(),
                       f'D:/Study/Module/Master Thesis/trained_models/Alexnet_cifar10_epoch_{epoch + 1}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    # Save the trained model
    torch.save(model.state_dict(), 'D:/Study/Module/Master Thesis/trained_models/Alexnet_cifar10.pth')

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
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print('Finished Training')



if __name__ == '__main__':
    freeze_support()
    main()