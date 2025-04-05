import os
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


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


root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'

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


if __name__ == '__main__':
    # root = 'D:/Study/Module/Master Thesis/dataset/CIFAR10'
    # dataprep = Data_prep_224_normal_N(root)
    #
    # train_idx, val_idx, test_idx = dataprep.get_category_index(0)
    # train_loader, val_loader, test_loader = dataprep.create_loaders(100, 2)

    # print(train_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    create_cifar224_gradcam()
