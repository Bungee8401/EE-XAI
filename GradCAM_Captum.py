import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from CustomDataset import Data_prep_224_normal_N
import requests
from torchvision.models.feature_extraction import create_feature_extractor
import pytorch_lightning
from torchvision.transforms import ToPILImage
from Vgg16bn_early_exit_small_fc import BranchVGG16BN
from matplotlib.colors import LinearSegmentedColormap


if __name__ == '__main__':
    # pytorch_lightning.seed_everything(2024)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet-50
    model = models.resnet50(pretrained=True).to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.eval()


    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=1, num_workers=8)

    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            idx += 1
            if idx > 2:
                break

            # main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)
            main_out = model(images)

            _, predicted_main = torch.max(main_out.data, 1)
            # _, predicted_exit1 = torch.max(exit1_out.data, 1)
            # _, predicted_exit2 = torch.max(exit2_out.data, 1)
            # _, predicted_exit3 = torch.max(exit3_out.data, 1)
            # _, predicted_exit4 = torch.max(exit4_out.data, 1)
            # _, predicted_exit5 = torch.max(exit5_out.data, 1)


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

            _ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
                                         sign="all",
                                         title="Layer 4 Block 2 Conv 3")

            upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, images.shape[2:])

            print(attributions_lgc.shape)
            print(upsamp_attr_lgc.shape)
            print(images.shape, torch.max(images))

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



# 1. Load the pre-trained ResNet-50 model
# 2. test this code with original image
