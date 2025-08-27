import os
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
from collections import defaultdict

trans_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_train_32 = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 224x224
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_test_32 = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_train224_01 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
])

trans_test224_01 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
])

def save_data(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_data(filename):
    with open(filename, "rb") as file:
        loaded_data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return loaded_data

def create_dataset(root, transform_training=trans_train, transform_test=trans_test):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def create_dataset_32(root, transform_training=trans_train_32, transform_test=trans_test_32):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def create_dataset_224_01(root, transform_training=trans_train224_01, transform_test=trans_test224_01):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def create_dataset_224_N(root, transform_training=trans_test, transform_test=trans_test):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def get_same_category_index(dataset, category):
    if isinstance(dataset, Subset):
        idx = torch.where(torch.tensor(dataset.dataset.targets)[dataset.indices] == category)[0].tolist()
    else:
        idx = torch.where(torch.tensor(dataset.targets) == category)[0].tolist()
    return idx

def img_norm(img):
    min_val = torch.min(img)
    max_val = torch.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

def grad_cam_mask(label, images):

    resnet_for_cam = ResNet_50().to(device)
    resnet_for_cam = nn.DataParallel(resnet_for_cam, device_ids=[0, 1, 2, 3])
    resnet_for_cam.load_state_dict(torch.load(r"weights/Resnet50/Resnet50_ori_epoch_20.pth", weights_only=True))
    resnet_for_cam.eval()

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

def load_cifar100(batch_size, num_workers, class_idx):
    import os
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset, random_split
    import numpy as np

    # Define transformations for CIFAR100
    trans_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trans_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Create directory if it doesn't exist
    os.makedirs("./CIFAR100", exist_ok=True)

    # Load the full CIFAR100 dataset
    train_set = datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=None)
    test_set = datasets.CIFAR100(root='./CIFAR100', train=False, download=True, transform=None)

    # Filter by class if class_idx is provided - much faster way using numpy operations
    if class_idx is not None:
        if not (0 <= class_idx <= 99):
            raise ValueError("class_idx must be between 0 and 99")

        # Get class indices using numpy for speed
        train_indices = np.where(np.array(train_set.targets) == class_idx)[0].tolist()
        test_indices = np.where(np.array(test_set.targets) == class_idx)[0].tolist()

        print(f"Filtered to class {class_idx}: {len(train_indices)} training, {len(test_indices)} test samples")
    else:
        # Use all indices if no class filtering
        train_indices = list(range(len(train_set)))
        test_indices = list(range(len(test_set)))

    # Apply transforms after filtering
    train_set.transform = trans_train
    test_set.transform = trans_test

    # Create subsets with the filtered indices
    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)

    # Split training set into training and validation (90/10)
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    trainset, valset = random_split(train_set, [train_size, val_size])

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"CIFAR100 dataset loaded: {len(trainset)} training, {len(valset)} validation, {len(test_set)} test samples")
    print(f"Train batch count: {len(trainloader)}, Val batch count: {len(valloader)}, Test batch count: {len(testloader)}")

    return trainloader, valloader, testloader


def load_cifar100_generator(batch_size, num_workers, class_idx):
    import os
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset, random_split
    import numpy as np

    # Define transformations for CIFAR100
    trans_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Create directory if it doesn't exist
    os.makedirs('./CIFAR100', exist_ok=True)

    # Load the full CIFAR100 dataset
    train_set = datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=None)
    test_set = datasets.CIFAR100(root='./CIFAR100', train=False, download=True, transform=None)


    # Filter by class if class_idx is provided - much faster way using numpy operations
    if class_idx is not None:
        if not (0 <= class_idx <= 99):
            raise ValueError("class_idx must be between 0 and 99")

        # Get class indices using numpy for speed
        train_indices = np.where(np.array(train_set.targets) == class_idx)[0].tolist()
        test_indices = np.where(np.array(test_set.targets) == class_idx)[0].tolist()

        print(f"Filtered to class {class_idx}: {len(train_indices)} training, {len(test_indices)} test samples")
    else:
        # Use all indices if no class filtering
        train_indices = list(range(len(train_set)))
        test_indices = list(range(len(test_set)))

    # Apply transforms after filtering
    train_set.transform = trans_test
    test_set.transform = trans_test

    # Create subsets with the filtered indices
    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)

    # Split training set into training and validation (90/10)
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    trainset, valset = random_split(train_set, [train_size, val_size])

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"CIFAR100 dataset loaded: {len(trainset)} training, {len(valset)} validation, {len(test_set)} test samples")
    print(f"Train batch count: {len(trainloader)}, Val batch count: {len(valloader)}, Test batch count: {len(testloader)}")

    return trainloader, valloader, testloader


class Data_prep_224_normal_N:
    def __init__(self, root):
        train_set, self.testset = create_dataset(root)

        split_path_train = "data_split/CIFAR224_train.pkl"
        split_path_val = "data_split/CIFAR224_val.pkl"
        if os.path.exists("./" + split_path_train) and os.path.exists("./" + split_path_val):
            train_indices, val_indices = (
                load_data("./" + split_path_train),
                load_data("./" + split_path_val),
            )
            self.trainset = Subset(train_set, train_indices)
            self.valset = Subset(train_set, val_indices)
        else:
            self.trainset, self.valset = random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])
            save_data("./" + split_path_train, self.trainset.indices)
            save_data("./" + split_path_val, self.valset.indices)
        self.get_same_category_index = get_same_category_index
        self.class_num = 10
        print("Create DataPrep for CIFAR224")

    # @staticmethod
    def create_loaders(self, batch_size, num_workers):

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return trainloader, valloader, testloader

    def get_category_index(self, category):
        train_idx = self.get_same_category_index(self.trainset, category)
        val_idx = self.get_same_category_index(self.valset, category)
        test_idx = self.get_same_category_index(self.testset, category)
        return train_idx, val_idx, test_idx

    def create_catogery_loaders(self, batch_size, num_workers, train_idx, val_idx, test_idx):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)

        return trainloader, valloader, testloader

class Data_prep_32_N:
    def __init__(self, root):
        train_set, self.testset = create_dataset_32(root)

        split_path_train = "data_split/CIFAR32_train.pkl"
        split_path_val = "data_split/CIFAR32_val.pkl"
        if os.path.exists("./" + split_path_train) and os.path.exists("./" + split_path_val):
            train_indices, val_indices = (
                load_data("./" + split_path_train),
                load_data("./" + split_path_val),
            )
            self.trainset = Subset(train_set, train_indices)
            self.valset = Subset(train_set, val_indices)
        else:
            self.trainset, self.valset = random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])
            save_data("./" + split_path_train, self.trainset.indices)
            save_data("./" + split_path_val, self.valset.indices)
        self.get_same_category_index = get_same_category_index
        self.class_num = 10
        print("Create DataPrep for CIFAR32")

    # @staticmethod
    def create_loaders(self, batch_size, num_workers):

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return trainloader, valloader, testloader

    def get_category_index(self, category):
        train_idx = self.get_same_category_index(self.trainset, category)
        val_idx = self.get_same_category_index(self.valset, category)
        test_idx = self.get_same_category_index(self.testset, category)
        return train_idx, val_idx, test_idx

    def create_catogery_loaders(self, batch_size, num_workers, train_idx, val_idx, test_idx):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)

        return trainloader, valloader, testloader

class Data_prep_224_01_N:
    def __init__(self, root):
        train_set, self.testset = create_dataset_224_01(root)

        split_path_train = "data_split/CIFAR224_01_train.pkl"
        split_path_val = "data_split/CIFAR224_01_val.pkl"
        if os.path.exists("./" + split_path_train) and os.path.exists("./" + split_path_val):
            train_indices, val_indices = (
                load_data("./" + split_path_train),
                load_data("./" + split_path_val),
            )
            self.trainset = Subset(train_set, train_indices)
            self.valset = Subset(train_set, val_indices)
        else:
            self.trainset, self.valset = random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])
            save_data("./" + split_path_train, self.trainset.indices)
            save_data("./" + split_path_val, self.valset.indices)
        self.get_same_category_index = get_same_category_index
        self.class_num = 10
        print("Create DataPrep for CIFAR224_01")

    # @staticmethod
    def create_loaders(self, batch_size, num_workers):

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return trainloader, valloader, testloader

    def get_category_index(self, category):
        train_idx = self.get_same_category_index(self.trainset, category)
        val_idx = self.get_same_category_index(self.valset, category)
        test_idx = self.get_same_category_index(self.testset, category)
        return train_idx, val_idx, test_idx

    def create_catogery_loaders(self, batch_size, num_workers, train_idx, val_idx, test_idx):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)

        return trainloader, valloader, testloader

class Data_prep_224_gen:
    def __init__(self, root):
        train_set, self.testset = create_dataset_224_N(root)

        split_path_train = "data_split/CIFAR224_gen_train.pkl"
        split_path_val = "data_split/CIFAR224_gen_val.pkl"
        if os.path.exists("./" + split_path_train) and os.path.exists("./" + split_path_val):
            train_indices, val_indices = (
                load_data("./" + split_path_train),
                load_data("./" + split_path_val),
            )
            self.trainset = Subset(train_set, train_indices)
            self.valset = Subset(train_set, val_indices)
        else:
            self.trainset, self.valset = random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])
            save_data("./" + split_path_train, self.trainset.indices)
            save_data("./" + split_path_val, self.valset.indices)
        self.get_same_category_index = get_same_category_index
        self.class_num = 10
        print("Create DataPrep for CIFAR224")

    # @staticmethod
    def create_loaders(self, batch_size, num_workers):

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return trainloader, valloader, testloader

    def get_category_index(self, category):
        train_idx = self.get_same_category_index(self.trainset, category)
        val_idx = self.get_same_category_index(self.valset, category)
        test_idx = self.get_same_category_index(self.testset, category)
        return train_idx, val_idx, test_idx

    def create_catogery_loaders(self, batch_size, num_workers, train_idx, val_idx, test_idx):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)

        return trainloader, valloader, testloader

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


class MaskedDataset(Dataset):
    def __init__(self, file_path):
        """
        Initialize the MaskedDataset with a file path to load masked data from.

        Args:
            file_path (str): Path to the pickle file containing masked images
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset file {file_path} not found. Create it using create_cifar224_gradcam function first.")

        self.data = load_data(file_path)
        self.images = self.data['images']
        self.labels = self.data['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def create_cifar224_gradcam():
    """
    Create CIFAR224_GRADCAM datasets for all categories using GradCAM.
    Processes train, validation and test splits and stores them as pickle files.
    """
    print("Creating CIFAR224_GRADCAM datasets for all categories")

    # Initialize data preparation
    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_gen(root)

    # Get general dataloaders
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=128, num_workers=2)

    # train_idx, val_idx, test_idx = dataprep.get_category_index(category=0)  # 0 airplane, 3 cat, 8 ship
    # # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
    # trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=128, num_workers=2,
    #                                                                       train_idx=train_idx, val_idx=val_idx,
    #                                                                       test_idx=test_idx)

    # Process each dataset split
    for split_name, dataloader in [
        ('train', trainloader),
        ('val', valloader),
        ('test', testloader)
    ]:
        print(f"Processing {split_name} set with {len(dataloader)} images")

        # Containers for masked images and labels
        all_masked_images = []
        all_labels = []

        for images, labels in dataloader:
            # Apply GradCAM masking
            images = images.to(device)
            labels = labels.to(device)

            masked_images = grad_cam_mask(labels, images)

            # Store the results
            all_masked_images.append(masked_images.cpu())
            all_labels.append(labels.cpu())

        # Concatenate all batches
        all_masked_images = torch.cat(all_masked_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Save to pickle file
        output_file = f'data_split/masked_CIFAR224_{split_name}.pkl'

        dataset_dict = {
            'images': all_masked_images.numpy(),
            'labels': all_labels.numpy()
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset_dict, f)

        print(f"Saved {len(all_labels)} masked images to {output_file}")

    print("CIFAR224_GRADCAM dataset creation complete")


root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'


# CIFAR-100 superclass mapping
CIFAR100_SUPERCLASS_MAPPING = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]  # vehicles 2
}

# Superclass names
SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man_made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_sized_mammals', 'non_insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]


class CIFAR100SuperclassDataset(Dataset):
    """Custom dataset for CIFAR-100 superclass data"""

    def __init__(self, original_dataset, superclass_id, fine_to_coarse_mapping):
        self.original_dataset = original_dataset
        self.superclass_id = superclass_id
        self.fine_classes = CIFAR100_SUPERCLASS_MAPPING[superclass_id]

        # Find indices of samples belonging to this superclass
        self.indices = []
        for idx, (_, label) in enumerate(original_dataset):
            if label in self.fine_classes:
                self.indices.append(idx)

        # Create label mapping for this superclass (0 to 4)
        self.label_mapping = {fine_class: i for i, fine_class in enumerate(self.fine_classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, original_label = self.original_dataset[original_idx]
        # Map to superclass-specific label (0-4)
        new_label = self.label_mapping[original_label]
        return image, new_label


def create_cifar100_superclass_dataloaders(data_dir='./data', superclass_id=0, batch_size=32,
                                           num_workers=2, train=True, shuffle=True):
    """
    Create dataloaders for each CIFAR-100 superclass

    Args:
        data_dir: Directory to store/load CIFAR-100 data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train: Whether to load training or test data
        shuffle: Whether to shuffle the data

    Returns:
        Dictionary mapping superclass names to their dataloaders
    """

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    # Load CIFAR-100 dataset
    cifar100_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)

    # Create fine-to-coarse mapping
    fine_to_coarse = {}
    for coarse_id, fine_classes in CIFAR100_SUPERCLASS_MAPPING.items():
        for fine_class in fine_classes:
            fine_to_coarse[fine_class] = coarse_id

    superclass_name = SUPERCLASS_NAMES[superclass_id]

    # Create superclass dataset
    superclass_dataset = CIFAR100SuperclassDataset(
        cifar100_dataset, superclass_id, fine_to_coarse
    )

    # Create dataloader
    dataloader = DataLoader(
        superclass_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


    print(f"Created dataloader for '{superclass_name}' with {len(superclass_dataset)} samples")

    return dataloader


def get_superclass_info():
    """Get information about CIFAR-100 superclasses"""
    info = {}
    for superclass_id, superclass_name in enumerate(SUPERCLASS_NAMES):
        fine_classes = CIFAR100_SUPERCLASS_MAPPING[superclass_id]
        info[superclass_name] = {
            'superclass_id': superclass_id,
            'fine_classes': fine_classes,
            'num_fine_classes': len(fine_classes)
        }
    return info


class CIFAR100CoarseDataset(Dataset):
    """Custom dataset for CIFAR-100 with coarse (superclass) labels"""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

        # Create fine-to-coarse mapping
        self.fine_to_coarse = {}
        for coarse_id, fine_classes in CIFAR100_SUPERCLASS_MAPPING.items():
            for fine_class in fine_classes:
                self.fine_to_coarse[fine_class] = coarse_id

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, fine_label = self.original_dataset[idx]
        # Map fine label to coarse label (0-19)
        coarse_label = self.fine_to_coarse[fine_label]
        return image, coarse_label


def create_cifar100_coarse_dataloaders(data_dir='./data', batch_size=32,
                                       num_workers=2, shuffle_train=True):
    """
    Create three dataloaders (train, val, test) for CIFAR-100 with coarse labels (20 superclasses)

    Args:
        data_dir: Directory to store/load CIFAR-100 data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        - train_loader: Training set (80% of original train set)
        - val_loader: Validation set (20% of original train set)
        - test_loader: Test set (original test set)
    """

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    # Load CIFAR-100 datasets
    cifar100_train_raw = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    cifar100_test_raw = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Create coarse label datasets
    cifar100_train = CIFAR100CoarseDataset(cifar100_train_raw)
    cifar100_test = CIFAR100CoarseDataset(cifar100_test_raw)

    # Split training set into train and validation (80/20 split)
    train_size = int(0.9 * len(cifar100_train))
    val_size = len(cifar100_train) - train_size

    # Create stratified split to ensure balanced classes
    train_indices, val_indices = create_stratified_split(
        cifar100_train, train_size, val_size
    )

    train_dataset = Subset(cifar100_train, train_indices)
    val_dataset = Subset(cifar100_train, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        cifar100_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Created CIFAR-100 coarse dataloaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(cifar100_test)} samples")
    print(f"  Total superclasses: 20")

    return train_loader, val_loader, test_loader


def create_stratified_split(dataset, train_size, val_size):
    """Create stratified split ensuring balanced classes in train and validation sets"""

    # Get all labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)

    labels = np.array(labels)

    # Create stratified split
    train_indices = []
    val_indices = []

    for class_id in range(20):  # 20 superclasses
        class_indices = np.where(labels == class_id)[0]
        class_train_size = int(len(class_indices) * 0.9)

        # Shuffle indices for this class
        np.random.shuffle(class_indices)

        train_indices.extend(class_indices[:class_train_size])
        val_indices.extend(class_indices[class_train_size:])

    return train_indices, val_indices


def get_coarse_class_names():
    """Get the names of all 20 superclasses"""
    return SUPERCLASS_NAMES

# Example usage
if __name__ == "__main__":
    # Create train dataloaders for all superclasses
    train_dataloader = create_cifar100_superclass_dataloaders(
        data_dir='./CIFAR100',
        superclass_id=0,
        batch_size=64,
        num_workers=4,
        train=True,
        shuffle=True
    )

    # Create test dataloaders for all superclasses
    test_dataloader = create_cifar100_superclass_dataloaders(
        data_dir='./CIFAR100',
        superclass_id=0,
        batch_size=64,
        num_workers=4,
        train=False,
        shuffle=False
    )

    print(f"\nExample: 'aquatic_mammals' superclass")
    print(f"Number of batches: {len(train_dataloader)}")

    # Get first batch
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels: {labels[:10]}")
        break

    # Print superclass information
    print("\nSuperclass Information:")
    superclass_info = get_superclass_info()
    for name, info in superclass_info.items():
        print(f"{name}: {info['num_fine_classes']} fine classes, "
              f"fine class IDs: {info['fine_classes']}")






