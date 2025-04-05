import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


def load_data(file_path):
    """Load data from a pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def display_masked_dataset(train_path, test_path, num_images=5):
    """Display images from masked datasets"""
    # Load datasets
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

    # Create a single figure for both train and test
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    # Display train images
    for i in range(num_images):
        if i < len(train_data):
            img, label = train_data["images"][i], train_data["labels"][i]
            img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Train - Label: {label}")
            axes[0, i].axis('off')

    # Display test images in the second row
    for i in range(num_images):
        if i < len(test_data):
            img, label = test_data["images"][i], test_data["labels"][i]
            img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC

            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Test - Label: {label}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def display_using_dataloader(train_path, test_path, num_images=5, batch_size=10):
    """Display images from masked datasets using DataLoader"""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # Load datasets
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    print(f"Train dataset size: {len(train_data['labels'])}")
    print(f"Test dataset size: {len(test_data['labels'])}")

    # Convert dictionary data to tensors
    train_images = torch.tensor(train_data["images"])
    train_labels = torch.tensor(train_data["labels"])
    test_images = torch.tensor(test_data["images"])
    test_labels = torch.tensor(test_data["labels"])

    # Create TensorDatasets (built-in PyTorch dataset)
    train_tensor_dataset = TensorDataset(train_images, train_labels)
    test_tensor_dataset = TensorDataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=True)

    # Get the first batch from each loader
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_images, train_labels = train_batch
    test_images, test_labels = test_batch

    # Display images in a single figure
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    # Display train images
    for i in range(num_images):
        if i < len(train_images):
            img = train_images[i].numpy()
            label = train_labels[i].item()
            img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Train - Label: {label}")
            axes[0, i].axis('off')

    # Display test images
    for i in range(num_images):
        if i < len(test_images):
            img = test_images[i].numpy()
            label = test_labels[i].item()
            img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC

            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Test - Label: {label}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_dataset_statistics(data_path):
    """Calculate mean and standard deviation of a dataset using DataLoader"""
    # Load dataset
    data = load_data(data_path)

    # Convert to tensor
    images = torch.tensor(data["images"], dtype=torch.float32)

    # Images are in format (N, C, H, W)
    # Calculate mean and std across all images for each channel
    mean = images.mean(dim=(0, 2, 3))
    std = images.std(dim=(0, 2, 3))

    print(f"Dataset: {data_path}")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")

    return mean, std

# Example usage
if __name__ == "__main__":
    category = 0  # 0 - Airplane (adjust based on your needs)

    # Paths to your masked dataset files
    train_path = f'data_split/masked_CIFAR224_train.pkl'
    test_path = f'data_split/masked_CIFAR224_test.pkl'

    # Display images
    # display_masked_dataset(train_path, test_path, num_images=5)

    # for i in range(10):
    #     display_using_dataloader(train_path, test_path, num_images=5, batch_size=10)

    # unit8 mean std
    mean, std = calculate_dataset_statistics(train_path)
    # Convert to 0-1 range
    normalized_mean = [x / 255.0 for x in mean]  # ~[0.2300, 0.2236, 0.2215]
    normalized_std = [x / 255.0 for x in std]  # ~[0.2184, 0.2097, 0.2118]

    print(normalized_mean, normalized_std)

    # Mean: [58.64394760131836, 57.030235290527344, 56.49018859863281]
    # Std: [55.69501876831055, 53.4670524597168, 54.009254455566406]
    # [0.2300, 0.2236, 0.2215]
    # [0.2184, 0.2097, 0.2118]

