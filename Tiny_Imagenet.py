import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import urllib.request
import zipfile
from PIL import Image
from tqdm import tqdm
from torchvision.models import ViT_B_16_Weights

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, annotations_file, class_to_idx, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        with open(annotations_file, 'r') as f:
            for line in f:
                img, wnid = line.strip().split('\t')[:2]
                if wnid in class_to_idx:
                    self.samples.append((img, class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.val_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Download Tiny ImageNet if not present
def download_and_extract_tiny_imagenet(data_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")
    if not os.path.exists(extract_path):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done.")
    else:
        print("Tiny ImageNet already downloaded and extracted.")

data_root = "Tiny Imagenet"
os.makedirs(data_root, exist_ok=True)
download_and_extract_tiny_imagenet(data_root)

# Resize images to 224x224 for ViT input
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jittering
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Update these paths to your Tiny ImageNet dataset
train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")
val_dir = os.path.join(data_root, "tiny-imagenet-200", "val", "images")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

# Build class_to_idx mapping from training set
class_to_idx = train_dataset.class_to_idx
val_annotations_file = os.path.join(data_root, "tiny-imagenet-200", "val", "val_annotations.txt")
val_images_dir = os.path.join(data_root, "tiny-imagenet-200", "val", "images")
val_dataset = TinyImageNetValDataset(val_images_dir, val_annotations_file, class_to_idx, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Logging utility
def log_message(message):
    log_dir = "running logs"
    log_file = os.path.join(log_dir, "vitb16_running log")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Use
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
model = models.vit_b_16(weights=vit_weights)
model.heads = nn.Linear(model.hidden_dim, 200)  # Tiny ImageNet has 200 classes
model = model.to(device)
print(model)

# Optionally freeze early layers (first 8 blocks)
for name, param in model.named_parameters():
    if 'blocks.0.' in name or 'blocks.1.' in name or 'blocks.2.' in name or 'blocks.3.' in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()

# Use AdamW instead of Adam
optimizer = optim.AdamW(
    [{'params': [p for n, p in model.named_parameters() if 'heads' not in n], 'lr': 1e-5},
            {'params': model.heads.parameters(), 'lr': 1e-4}],
    weight_decay=0.01)

# Cosine Annealing LR Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training loop
num_epochs = 100
best_val_acc = 0.0  # Track best validation accuracy

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total if total > 0 else 0
        train_loader_tqdm.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
    train_acc = 100. * correct / total
    train_log = f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.2f}%"
    print('\n', train_log)
    log_message(train_log)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            val_loader_tqdm.set_postfix(acc=f"{val_acc:.2f}%")
    val_acc = 100. * val_correct / val_total
    val_log = f"Epoch {epoch+1}/{num_epochs}, Validation Acc: {val_acc:.2f}%"
    print('\n', val_log)
    log_message(val_log)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "weights/vitb16_tiny_imagenet_best.pth")
        log_message(f"Best model saved at epoch {epoch+1} with val acc: {best_val_acc:.2f}%")

    # Step the scheduler
    scheduler.step()

# Save the fine-tuned model
torch.save(model.state_dict(), "weights/vitb16_tiny_imagenet.pth")
