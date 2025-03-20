import torch
import torch.nn as nn
import torchvision.models as models
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import numpy as np
from CustomDataset import Data_prep_224_normal_N
import pytorch_lightning
from matplotlib import pyplot as plt
import torch.optim as optim

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes

        # Load the pretrained ResNet50 model
        resnet = models.resnet50(weights=True)

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

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            main_out = model(images)
            _, predicted = torch.max(main_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def fine_tune(num_epochs):
    # Only train the classifier parameters
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}")

        test()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       f'training_weights/resnet50/Resnet50_ori_epoch_{epoch + 1}.pth')

    print("Finished fine-tuning")

# captum package

def mask():
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            idx += 1
            if idx > 2:
                break

            main_out = model(images)

            _, predicted_main = torch.max(main_out.data, 1)

            # 1. Integrated Gradients
            # # Initialize the attribution algorithm with the model
            # integrated_gradients = IntegratedGradients(model)
            #
            # # Ask the algorithm to attribute our output target to
            # attributions_ig = integrated_gradients.attribute(images, target=predicted_main, n_steps=200)
            #
            # # Show the original image for comparison
            # _ = viz.visualize_image_attr(None, np.transpose(images.squeeze().cpu().detach().numpy(), (1,2,0)),
            #                       method="original_image", title="Original Image")
            #
            # default_cmap = LinearSegmentedColormap.from_list('custom blue',
            #                                                  [(0, '#ffffff'),
            #                                                   (0.25, '#0000ff'),
            #                                                   (1, '#0000ff')], N=256)
            #
            # _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
            #                              np.transpose(images.squeeze().cpu().detach().numpy(), (1,2,0)),
            #                              method='heat_map',
            #                              cmap=default_cmap,
            #                              show_colorbar=True,
            #                              sign='positive',
            #                              title='Integrated Gradients')

            # 2. Layer GradCam

            layer_gradcam = LayerGradCam(model, model.module.layer4[2].conv3)
            attributions_lgc = layer_gradcam.attribute(images, target=predicted_main)

            upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, images.shape[2:])

            _ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
                                         sign="all",
                                         title="Layer 4 Block 2 Conv 3")

            # print(attributions_lgc.shape)
            print(upsamp_attr_lgc.shape)
            print(upsamp_attr_lgc)
            # print(images.shape, torch.max(images))

            min_val = torch.min(images)
            max_val = torch.max(images)
            images = (images - min_val) / (max_val - min_val)

            _ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
                                                  images[0].cpu().permute(1, 2, 0).numpy(),
                                                  ["original_image", "blended_heat_map", "masked_image"],
                                                  ["all", "positive", "positive"],
                                                  show_colorbar=True,
                                                  titles=["Original", "Positive Attribution", "Masked"],
                                                  fig_size=(18, 6))

            # masked_image = viz.visualize_image_attr(upsamp_attr_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
            #                                       images[0].cpu().permute(1, 2, 0).numpy(),
            #                                       "masked_image",
            #                                       "positive",
            #                                       show_colorbar=True,
            #                                       title="Masked",
            #                                       fig_size=(18, 6))

def grad_cam_mask(label, images):

    # layer_gradcam = LayerGradCam(classifier, classifier.module.layer4[2].conv3) # Resnet50
    # def wrapped_model(inp):
    #     return classifier(inp)[0]

    layer_gradcam = LayerGradCam(model, model.module.layer4[2].conv3) # VGG16
    attributions_lgc = layer_gradcam.attribute(images, target=label)
    upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, images.shape[2:])
    # upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, images.shape[2:], interpolate_mode=)


    return upsamp_attr_lgc

def test_grad_mask():
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            masks = grad_cam_mask(labels, images)

            if idx > 2:
                break

            for i in range(images.size(0)):
                if i > 10:
                    break
                img = images[i].cpu().permute(1, 2, 0).numpy()
                mask = masks[i].cpu().permute(1, 2, 0).numpy()
                masked_img = img * mask

                img = img_norm(img)
                masked_img = img_norm(masked_img)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(img)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Masked Image")
                plt.imshow(masked_img)
                plt.axis('off')

                plt.show()

def img_norm(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img


# putorch_grad_cam package
def gradcam_visualization():
    # Define target layer for ResNet50
    target_layers = [model.module.layer4[-1]]
    batch_size = 32  # Set your desired batch size

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    for batch_idx, (images, labels) in enumerate(testloader):
        if batch_idx > 2:  # Process first 3 batches only
            break

        images = images.to(device)
        labels = labels.to(device)

        # Process each image in the batch
        for idx in range(images.shape[0]):
            input_tensor = images[idx].unsqueeze(0)  # Add batch dimension back
            target = ClassifierOutputTarget(labels[idx].item())

            # Generate GradCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
            grayscale_cam = grayscale_cam[0]  # Remove batch dimension

            # Convert image for visualization
            rgb_img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

            # Create masked image by multiplying original with GradCAM
            masked_img = rgb_img * grayscale_cam[..., None]
            masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min())

            # Overlay GradCAM on image
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Display results
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title(f'Original Image (Batch {batch_idx}, Image {idx})')
            plt.imshow(rgb_img)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('GradCAM')
            plt.imshow(visualization)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Masked Image')
            plt.imshow(masked_img)
            plt.axis('off')

            plt.show()



if __name__ == '__main__':
    pytorch_lightning.seed_everything(2024)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet-50
    model = ResNet50().to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(torch.load(r"weights/Resnet50/Resnet50_ori_epoch_20.pth", weights_only=True))

    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=2, num_workers=8)

    # fine_tune(50)
    # test()
    # test_grad_mask()

    # use pytorch_grad_cam package to compute the gradcam and visualize the results
    gradcam_visualization()



