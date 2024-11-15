import torch
from torchvision.models import vgg16_bn

# Initialize the VGG16 model with batch normalization
model = vgg16_bn(weights=None)

# Print the model architecture
print(model)