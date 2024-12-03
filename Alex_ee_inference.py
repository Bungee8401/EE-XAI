import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import alexnet
import matplotlib.pyplot as plt

from Alexnet_early_exit import BranchedAlexNet

# class BranchedAlexNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(BranchedAlexNet, self).__init__()
#         # Load pretrained AlexNet
#         original_model = alexnet(weights=None)
#
#         # Modify the first layer for CIFAR-10 input (32x32x3)
#         original_model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#
#         # Main branch (full AlexNet)
#         self.features = original_model.features
#         self.avgpool = original_model.avgpool
#         self.classifier = original_model.classifier
#         self.classifier[6] = nn.Linear(4096, num_classes)
#
#         # Branch 1 after 5th layer
#         self.branch1 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(43200, num_classes)
#         )
#
#         # Branch 2 after 9th layer
#         self.branch2 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(12544, num_classes)
#         )
#
#     def forward(self, x):
#         # Main branch
#         x_main = self.features(x)
#         x_main = self.avgpool(x_main)
#         x_main = torch.flatten(x_main, 1)
#         out_main = self.classifier(x_main)
#
#         # Branch 1 (After 5nd layer)
#         x_branch1 = self.features[:5](x)  # Output after the 2nd layer
#         out_branch1 = self.branch1(x_branch1)
#
#         # Branch 2 (After 9th layer)
#         x_branch2 = self.features[:9](x)  # Output after the 5th layer
#         out_branch2 = self.branch2(x_branch2)
#
#         return out_main, out_branch1, out_branch2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)


# Load the trained model
model = BranchedAlexNet(num_classes=10).to(device)
model.load_state_dict(torch.load(r"D:\Study\Module\Master Thesis\trained_models\B-Alex lr=0.001 transfer learning\B-Alex_cifar10_epoch_30.pth",
                                 weights_only=True))
model.eval()

# Define the inference function
def threshold_inference(model, dataloader, exit_thresholds):

    correct_main = 0
    correct_exit1 = 0
    correct_exit2 = 0
    correct_exit3 = 0
    correct_exit4 = 0
    correct_exit5 = 0
    total = 0
    exit_counts = [0, 0, 0, 0, 0, 0]  # [exit1-5, main]

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate softmax and entropy for exit1-5
            softmax_exit1 = F.softmax(exit1_out, dim=1)
            entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

            # Calculate softmax and entropy for exit2
            softmax_exit2 = F.softmax(exit2_out, dim=1)
            entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

            # Calculate softmax and entropy for exit3
            softmax_exit3 = F.softmax(exit3_out, dim=1)
            entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

            # Calculate softmax and entropy for exit4
            softmax_exit4 = F.softmax(exit4_out, dim=1)
            entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

            # Calculate softmax and entropy for exit5
            softmax_exit5 = F.softmax(exit5_out, dim=1)
            entropy_exit5 = -torch.sum(softmax_exit5 * torch.log(softmax_exit5 + 1e-5), dim=1)

            # Determine exit points based on thresholds
            for i in range(labels.size(0)):
                if entropy_exit1[i] < exit_thresholds[0]:
                    _, predicted = torch.max(exit1_out[i].data, 0)
                    correct_exit1 += (predicted == labels[i]).item()
                    exit_counts[0] += 1
                elif entropy_exit2[i] < exit_thresholds[1]:
                    _, predicted = torch.max(exit2_out[i].data, 0)
                    correct_exit2 += (predicted == labels[i]).item()
                    exit_counts[1] += 1
                elif entropy_exit3[i] < exit_thresholds[2]:
                    _, predicted = torch.max(exit3_out[i].data, 0)
                    correct_exit3 += (predicted == labels[i]).item()
                    exit_counts[2] += 1
                elif entropy_exit4[i] < exit_thresholds[3]:
                    _, predicted = torch.max(exit4_out[i].data, 0)
                    correct_exit4 += (predicted == labels[i]).item()
                    exit_counts[3] += 1
                elif entropy_exit5[i] < exit_thresholds[4]:
                    _, predicted = torch.max(exit5_out[i].data, 0)
                    correct_exit5 += (predicted == labels[i]).item()
                    exit_counts[4] += 1
                else:
                    _, predicted = torch.max(main_out[i].data, 0)
                    correct_main += (predicted == labels[i]).item()
                    exit_counts[5] += 1

    # Calculate accuracy and exit ratios
    exit1_accuracy = 100 * correct_exit1 / exit_counts[0] if exit_counts[0] > 0 else 0
    exit2_accuracy = 100 * correct_exit2 / exit_counts[1] if exit_counts[1] > 0 else 0
    exit3_accuracy = 100 * correct_exit3 / exit_counts[2] if exit_counts[2] > 0 else 0
    exit4_accuracy = 100 * correct_exit4 / exit_counts[3] if exit_counts[3] > 0 else 0
    exit5_accuracy = 100 * correct_exit5 / exit_counts[4] if exit_counts[4] > 0 else 0
    main_accuracy = 100 * correct_main / exit_counts[5] if exit_counts[5] > 0 else 0

    exit_ratios = [100 * count / total for count in exit_counts]

    return (exit1_accuracy, exit2_accuracy, exit3_accuracy, exit4_accuracy, exit5_accuracy, main_accuracy), exit_ratios

if __name__ == '__main__':
    # Example usage
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', train=False,
                                       download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    exit1_thresholds = []
    exit2_thresholds = []
    exit3_thresholds = []
    exit4_thresholds = []
    exit5_thresholds = []

    exit1_accuracies = []
    exit2_accuracies = []
    exit3_accuracies = []
    exit4_accuracies = []
    exit5_accuracies = []

    main_accuracies = []

    exit1_ratios = []
    exit2_ratios = []
    exit3_ratios = []
    exit4_ratios = []
    exit5_ratios = []

    main_ratios = []

    exit_thresholds = [0.001, 0.001, 0.001, 0.001, 0.001]

    sweep_exit = 1000

    for i in range(sweep_exit):

        accuracy, exit_ratios = threshold_inference(model, testloader, exit_thresholds)

        exit1_thresholds.append(exit1_threshold)
        exit2_thresholds.append(exit2_threshold)

        exit1_accuracies.append(accuracy[0])
        exit2_accuracies.append(accuracy[1])
        exit3_accuracies.append(accuracy[2])
        exit4_accuracies.append(accuracy[3])
        exit5_accuracies.append(accuracy[4])

        main_accuracies.append(accuracy[5])

        exit1_ratios.append(exit_ratios[0])
        exit2_ratios.append(exit_ratios[1])
        exit3_ratios.append(exit_ratios[2])
        exit4_ratios.append(exit_ratios[3])
        exit5_ratios.append(exit_ratios[4])

        main_ratios.append(exit_ratios[2])

        exit1_threshold = exit1_threshold + 0.001
        exit2_threshold = exit2_threshold + 0.001

        print(f"exit_thresholds: {exit1_threshold}, exit2_threshold: {exit2_threshold}")
        print(f"Classification Accuracy: Exit1: {accuracy[0]:.2f}%, Exit2: {accuracy[1]:.2f}%, Main: {accuracy[2]:.2f}%")
        print(f"Exit Ratios: Exit1: {exit_ratios[0]:.2f}%, Exit2: {exit_ratios[1]:.2f}%, Main: {exit_ratios[2]:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(exit1_thresholds, exit1_accuracies, label='Exit1 Accuracy')
    plt.plot(exit2_thresholds, exit2_accuracies, label='Exit2 Accuracy')
    plt.plot(exit2_thresholds, main_accuracies, label='Main Accuracy')
    plt.xlabel('Exit Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Exit Threshold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(exit1_thresholds, exit1_ratios, label='Exit1 Ratio')
    plt.plot(exit2_thresholds, exit2_ratios, label='Exit2 Ratio')
    plt.plot(exit2_thresholds, main_ratios, label='Main Ratio')
    plt.xlabel('Exit Threshold')
    plt.ylabel('Exit Ratio (%)')
    plt.title('Exit Ratio vs Exit Threshold')
    plt.legend()

    plt.tight_layout()
    plt.show()

#TODO:  1. are thresholds independent of each other? -> can i sweep 1 threshold to find the best value? -> paper search
#       2. what is "best"? how to measure best in inference? -> paper search
#       3. how to stop later branches when early branches made decisions already? is this related to current work? -> EECE code, re-write data-loader possible

