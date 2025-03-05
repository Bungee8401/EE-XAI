import torch
from torchvision import transforms
from PIL import Image
from Alexnet_early_exit import BranchedAlexNet
from Vgg16bn_early_exit_small_fc import BranchVGG16BN
from Resnet50_ee import BranchedResNet50

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

def test_image_with_classifiers(image_path, classifier_name):

    image = load_image(image_path).to(device)

    if classifier_name == 'B_alex':
        classifier = BranchedAlexNet()
        classifier.load_state_dict(torch.load('weights/B-Alex final/B-Alex_cifar10.pth', weights_only=True))
    elif classifier_name == 'B_Vgg16':
        classifier = BranchVGG16BN()
        classifier.load_state_dict(torch.load('weights/Vgg16bn_ee_224/Vgg16bn_epoch_15.pth', weights_only=True))
    elif classifier_name == 'B_Resnet50':
        classifier = BranchedResNet50()
        classifier.load_state_dict(torch.load('weights/Resnet50/B-Resnet50_epoch_10.pth', weights_only=True))

    classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        if classifier_name == 'B_alex':
            out_main, out_branch1, out_branch2, out_branch3, out_branch4, out_branch5 = classifier(image)
        elif classifier_name == 'B_Vgg16':
            out_main, out_branch1, out_branch2, out_branch3, out_branch4, out_branch5 = classifier(image)
        elif classifier_name == 'B_Resnet50':
            out_main, out_branch1, out_branch2, out_branch3, out_branch4 = classifier(image)

        _, predicted = torch.max(out_main, 1)
        print(f'Predicted class: {predicted.item()}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = 'Results/March07Reports/vgg16/white board test/plot_2025-03-04 16-12-52_0.png'  # Replace with the path to your local image
    classifier_name = 'B_Vgg16'  # Replace with the classifier you want to use
    test_image_with_classifiers(image_path, classifier_name)