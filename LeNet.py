import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support

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

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize the VGG16 model without pre-trained weights
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)  # Modify the last layer for 10 classes
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # List to store accuracy after each epoch
    epoch_acc = []

    # Training the model
    for epoch in range(25):  # Loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Evaluate the model after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        epoch_acc.append(accuracy)
        print(f'Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'D:/Study/Module/Master Thesis/trained_models/vgg16_cifar10.pth')

    # Plot accuracy-epoch diagram
    plt.plot(range(1, 25), epoch_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()