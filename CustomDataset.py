import os
import pickle
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

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

def create_dataset(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', transform_training=trans_train, transform_test=trans_test):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def create_dataset_32(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', transform_training=trans_train_32, transform_test=trans_test_32):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def create_dataset_224_01(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', transform_training=trans_train224_01, transform_test=trans_test224_01):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def get_same_category_index(dataset, category):
    if isinstance(dataset, Subset):
        idx = torch.where(torch.tensor(dataset.dataset.targets)[dataset.indices] == category)[0].tolist()
    else:
        idx = torch.where(torch.tensor(dataset.targets) == category)[0].tolist()
    return idx

root = 'D:/Study/Module/Master Thesis/dataset/CIFAR10'

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

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, sampler=train_sampler)
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

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, sampler=train_sampler)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)

        return trainloader, valloader, testloader

if __name__ == '__main__':
    # root = 'D:/Study/Module/Master Thesis/dataset/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)

    train_loader, val_loader, test_loader = dataprep.create_catogery_loaders(dataprep, 128)
    train_idx, val_idx, test_idx = dataprep.get_category_index(0)
    print(train_idx)