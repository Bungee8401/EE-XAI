import torch
import torch.nn.functional as F
import numpy as np
import cv2
from Alexnet_early_exit import BranchedAlexNet
import matplotlib.pyplot as plt
import numpy as np
from CustomDataset import Data_prep_224_gen
import pytorch_lightning
from torch.utils.data import DataLoader, TensorDataset
import pickle


def load_dataset(file_path):
    # Load the .pkl file
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    images, labels = data["images"], data["labels"]
    dataset = TensorDataset(images, labels)  # Create a TensorDataset
    return dataset

def convert_to_grayscale(input_img):
    # Weighted sum of R, G, and B channels (standard grayscale calculation)
    gray_img = 0.299 * input_img[..., 0] + 0.587 * input_img[..., 1] + 0.114 * input_img[..., 2]
    # Expand grayscale to 3 channels to match heatmap dimensions
    gray_img = np.stack([gray_img, gray_img, gray_img], axis=-1)
    return gray_img

def enhance_heatmap(heatmap):
    """
    热力图增强处理（对比度增强+锐化）
    """
    # 转换为uint8格式
    heatmap = np.uint8(255 * heatmap)

    # CLAHE对比度限制直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(heatmap)

    # 锐化处理
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened / 255.0

def visualize_gradcams(input_img, gradcams):
    """
    Visualize input image and Grad-CAM heatmaps
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Exit1', 'Exit2', 'Exit3', 'Exit4', 'Exit5', 'Original Image']

    # Preprocess input image for visualization
    input_img = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
    # input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(gradcams):
            ax.imshow(input_img)
            ax.set_title(titles[idx])
            break

        # **热力图增强处理**
        heatmap = gradcams[idx]

        # 3. 归一化到[0,1]
        heatmap = (heatmap - heatmap.min())
        heatmap /= (heatmap.max() - heatmap.min() + 1e-8)

        heatmap = np.transpose(heatmap, (1, 2, 0))

        # 1) simply truncate the heatmap
        heatmap = heatmap[..., :3]
        #todo: find better ways to collapse multiple channels to 3 channels

        # 2) copy the mean to 3 channels
        # collapsed = np.mean(heatmap, axis=-1)  # Shape: [H, W]
        # heatmap = np.stack([collapsed] * 3, axis=-1)  # Duplicate to 3 channels (RGB-like)

        # 3) keep the max and copy it to 3 Channels
        # # Max across channels
        # collapsed = np.max(heatmap, axis=-1)  # Shape: [H, W]
        # # Duplicate to create 3 channels
        # heatmap = np.stack([collapsed] * 3, axis=-1)

        # gray_img = convert_to_grayscale(input_img)


        # ax.imshow(input_img)
        ax.imshow(heatmap, alpha=1.0, cmap='jet')
        ax.set_title(titles[idx])
        # ax.axis('off')

    plt.tight_layout()
    plt.show()

def compute_gradcam(model, input_tensor, target_layer, exit_output, pred_class):
    """
    Compute Grad-CAM for the target layer during inference.
    """
    feature_map = None
    gradient = None

    # Hook functions to capture feature maps and gradients
    def forward_hook(module, inp, out):
        nonlocal feature_map
        feature_map = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].detach()

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()  # Set model to evaluation mode

    # Temporarily enable gradients for Grad-CAM
    with torch.enable_grad():

        # Ensure gradients are zeroed out
        model.zero_grad()
        # Forward pass
        output = model(input_tensor)

        # Set up one-hot encoding for the predicted class
        one_hot = torch.zeros_like(exit_output)
        one_hot[0][pred_class] = 1  # One-hot encoding for predicted class

        # exit_output.backward(gradient=one_hot, retain_graph=True)

        # Compute only the gradients we care about
        gradients_tuple = torch.autograd.grad(
            outputs=exit_output,
            inputs=target_layer.parameters(),
            grad_outputs=one_hot,
            create_graph=True,
            retain_graph=True,
        )

        # Ensure the gradient tensor is extracted from tuple
        gradient = gradients_tuple[0]  # Extract the first tensor, ie, the weights gradients, from the tuple
        gradient = gradient.permute(1, 0, 2, 3) # from (C,B,H,W) -> (B,C,H,W)

        # Verify if gradients are captured
        if gradient is None:
            raise ValueError("Gradient was not captured. Ensure backward hook is triggered.")

        # Compute Grad-CAM weights and map
        weights = gradient.mean(dim=(2, 3), keepdim=True) # spatially averaged gradients as a proxy for how much each channel's features contribute to the target class output.
        #todo:
        # 1. select top x% channels
        # 2. use top channels to optain weights
        # 3. then, gradcam

        gradcam = (weights * feature_map).sum(dim=1, keepdim=True)
        gradcam = F.relu(gradcam)

        # Upsample Grad-CAM to match input image size
        gradcam = F.interpolate(gradcam, input_tensor.shape[-2:], mode='bilinear', align_corners=False)

    # Remove hooks after computation
    forward_handle.remove()
    backward_handle.remove()

    return gradcam.squeeze().cpu().detach().numpy()

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BranchedAlexNet(num_classes=10).to(device)
    model.load_state_dict(
        torch.load(r"/home/yibo/PycharmProjects/Thesis/weights/B-Alex final/B-Alex_cifar10.pth", weights_only=True))
    model.eval()

    pytorch_lightning.seed_everything(2024)

    # # 1) load original cifar10
    # root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    # dataprep = Data_prep_224_gen(root)
    # train_idx, val_idx, test_idx = dataprep.get_category_index(category=0)  # 0 airplane, 1....
    # train_loader, val_loader, test_loader = dataprep.create_catogery_loaders(batch_size=256, num_workers=8,
    #                                                                       train_idx=train_idx, val_idx=val_idx,
    #                                                                       test_idx=test_idx)

    # 2) load gen_cifar10
    train_dataset = load_dataset('data_split/generated_CIFAR224_train.pkl')
    val_dataset = load_dataset('data_split/generated_CIFAR224_val.pkl')
    test_dataset = load_dataset('data_split/generated_CIFAR224_test.pkl')

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)


    images = []
    labels = []
    seq = 80
    for idx, data in enumerate(train_loader):
        img, label = data[0].to(device), data[1].to(device)
        if label[0] == 9:
            images.append(img[seq:seq+1])
            labels.append(label)

        if idx == 0:
            break


    images = torch.stack(images[:10]).squeeze(0)
    labels = torch.stack(labels[:10])
    input_tensor = images  # Example input

    # Forward pass to get outputs
    out_main, out_branch1, out_branch2, out_branch3, out_branch4, out_branch5 = model(input_tensor)
    features = model.gram_cam_features(input_tensor)
    exit_outputs = [out_branch1, out_branch2, out_branch3, out_branch4, out_branch5]

    # Get Grad-CAM target layers
    target_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]

    # Compute Grad-CAMs for all exits
    gradcams = []
    for exit_idx in range(len(exit_outputs)):
        # Select class prediction for gradient calculation
        pred_class = exit_outputs[exit_idx].argmax()

        # Compute Grad-CAM
        raw_gradcam = compute_gradcam(
            model,
            input_tensor,
            target_layers[exit_idx],  # Ensure this is the correct target layer
            exit_outputs[exit_idx],
            pred_class  # Pass the predicted class
        )
        # enhanced_cam = enhance_heatmap(raw_gradcam)
        # gradcams.append(enhanced_cam)
        gradcams.append(raw_gradcam)

    # Visualize Grad-CAM results
    visualize_gradcams(input_tensor, gradcams)
