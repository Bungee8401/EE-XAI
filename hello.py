import torch
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import Alex_ee_inference
import Alexnet_early_exit

# seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

# Define the transform to convert images to tensors
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize images to the size expected by AlexNet
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.ImageFolder(root='D:\Code\Thesis\Airplane_right\exit1', transform=transform)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(8, 12))
# Display the first image in the testset
image, label = testset[0]
image = image.permute(1, 2, 0)
image = image.numpy()
image = (image - image.min()) / (image.max() - image.min())
axs[0].imshow(image)
axs[0].set_title(f'Label: {label}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the size expected by AlexNet
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.ImageFolder(root='D:\Code\Thesis\Airplane_right\exit1', transform=transform)

# Display the first image in the testset
image, label = testset[0]
image = image.permute(1, 2, 0)
image = image.numpy()
image = (image - image.min()) / (image.max() - image.min())
axs[1].imshow(image)
axs[1].set_title(f'Label: {label}')

plt.show()
