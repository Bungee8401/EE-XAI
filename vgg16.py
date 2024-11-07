# prompt: vgg16 untrained model, train with cifar10, use pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import numpy as np

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='D:\Study\Module\Master Thesis\dataset\CIFAR10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 4

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Move images and labels to the GPU
images, labels = images.to(device), labels.to(device)

# Show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()  # move to CPU before converting to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# Show all images in the batch
imshow(torchvision.utils.make_grid(images[:batch_size]))  # move to CPU before showing
