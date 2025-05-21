import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Vgg16bn_early_exit_small_fc import BranchVGG16BN
import os
from CustomDataset import Data_prep_224_normal_N

# seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_lightning.seed_everything(2024)

# Load the trained model
model = BranchVGG16BN(num_classes=10).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(r"weights/Vgg16bn_ee_224/Vgg16bn_epoch_15.pth", weights_only=True))

def save_right_image(image, label, predicted, exit_name, cnt):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    os.makedirs(f'Alex_thresh_right/{exit_name}', exist_ok=True)
    torchvision.utils.save_image(image, f'Alex_thresh_right/{exit_name}/{class_names[label]}_{class_names[predicted]}_{cnt}.png')

def save_wrong_image(image, label, predicted, exit_name, cnt):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # label_folder = class_names[label]
    os.makedirs(f'Alex_thresh_wrong/{exit_name}', exist_ok=True)
    torchvision.utils.save_image(image, f'Alex_thresh_wrong/{exit_name}/{class_names[label]}_{class_names[predicted]}_{cnt}.png')

def simple_inference(model, dataloader):
    model.eval()
    correct_main = 0
    correct_exit1 = 0
    correct_exit2 = 0
    correct_exit3 = 0
    correct_exit4 = 0
    correct_exit5 = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate predictions for each exit
            _, predicted_main = torch.max(main_out.data, 1)
            _, predicted_exit1 = torch.max(exit1_out.data, 1)
            _, predicted_exit2 = torch.max(exit2_out.data, 1)
            _, predicted_exit3 = torch.max(exit3_out.data, 1)
            _, predicted_exit4 = torch.max(exit4_out.data, 1)
            _, predicted_exit5 = torch.max(exit5_out.data, 1)

            correct_main += (predicted_main == labels).sum().item()
            correct_exit1 += (predicted_exit1 == labels).sum().item()
            correct_exit2 += (predicted_exit2 == labels).sum().item()
            correct_exit3 += (predicted_exit3 == labels).sum().item()
            correct_exit4 += (predicted_exit4 == labels).sum().item()
            correct_exit5 += (predicted_exit5 == labels).sum().item()

    accuracy_main = 100 * correct_main / total
    accuracy_exit1 = 100 * correct_exit1 / total
    accuracy_exit2 = 100 * correct_exit2 / total
    accuracy_exit3 = 100 * correct_exit3 / total
    accuracy_exit4 = 100 * correct_exit4 / total
    accuracy_exit5 = 100 * correct_exit5 / total

    return [accuracy_exit1, accuracy_exit2, accuracy_exit3, accuracy_exit4, accuracy_exit5, accuracy_main]

def threshold_finder(model, dataloader, initial_thresholds, accuracy, step, tolerance):
    optimal_thresholds = initial_thresholds.copy()
    best_accuracies = [0] * len(initial_thresholds)
    best_exit_ratios = [0] * len(initial_thresholds)
    goal_accuracy = [acc - tolerance for acc in accuracy]
    upper_acc = accuracy[-1]

    for i in range(len(initial_thresholds)):
        print("----finding optimal threshold for early exit point----", i+1)
        while accuracy[i] < upper_acc  and 0 < optimal_thresholds[i] <= 2.3: # max entropy for 10 class problem; acc needs upper bound, otherwise programm will keep iterating to make acc higher as possible in the step range
            model.eval()
            accuracy, exit_ratios = threshold_inference(model, dataloader, optimal_thresholds)

            best_accuracies[i] = accuracy[i]
            best_exit_ratios[i] = exit_ratios[i]
            optimal_thresholds[i] -= step #exit ratio decrease from 100% to max%, with allowed 1% acc drop
            print(f"current thresh: {optimal_thresholds[i]}, Accuracy: {best_accuracies[i]}, Exit Ratio: {best_exit_ratios[i]}")
        # revert the last step and prevent >=2.3
        if optimal_thresholds[i] == 2.2:
            optimal_thresholds[i] += 0.05 # revert the last step cuz it's while loop
        else:
            optimal_thresholds[i] += step

        print(f"Optimal thresh: {optimal_thresholds[i]}, Accuracy: {best_accuracies[i]}, Exit Ratio: {best_exit_ratios[i]}")

        optimal_thresholds[i] = max(0.01, min(2.2, optimal_thresholds[i])) # prevent 0 and 1.0 thresholds

    return optimal_thresholds, best_accuracies, best_exit_ratios

def threshold_inference(model, dataloader, exit_thresholds):
    model.eval()

    correct_main = 0
    correct_exit1 = 0
    correct_exit2 = 0
    correct_exit3 = 0
    correct_exit4 = 0
    correct_exit5 = 0
    total = 0
    exit_counts = [0, 0, 0, 0, 0, 0]  # [exit1-5, main]
    cnt = 0

    # entropy-based criteria as in BranchyNet; in cifar10, max entropy is 2.3; smaller entropy, more confident
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate softmax and entropy for exit1-5
            softmax_exit1 = F.softmax(exit1_out, dim=1)
            entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

            softmax_exit2 = F.softmax(exit2_out, dim=1)
            entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

            softmax_exit3 = F.softmax(exit3_out, dim=1)
            entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

            softmax_exit4 = F.softmax(exit4_out, dim=1)
            entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

            softmax_exit5 = F.softmax(exit5_out, dim=1)
            entropy_exit5 = -torch.sum(softmax_exit5 * torch.log(softmax_exit5 + 1e-5), dim=1)

            # Determine exit points based on thresholds; if-elif-else ladder, skip rest if 1 exit point is found
            for i in range(labels.size(0)):
                if entropy_exit1[i] < exit_thresholds[0]:
                    _, predicted = torch.max(exit1_out[i].data, 0)
                    exit_counts[0] += 1
                    if predicted == labels[i]:
                        correct_exit1 += 1

                elif entropy_exit2[i] < exit_thresholds[1]:
                    _, predicted = torch.max(exit2_out[i].data, 0)
                    exit_counts[1] += 1
                    if predicted == labels[i]:
                        correct_exit2 += (predicted == labels[i]).item()

                elif entropy_exit3[i] < exit_thresholds[2]:
                    _, predicted = torch.max(exit3_out[i].data, 0)
                    exit_counts[2] += 1
                    if predicted == labels[i]:
                        correct_exit3 += (predicted == labels[i]).item()

                elif entropy_exit4[i] < exit_thresholds[3]:
                    _, predicted = torch.max(exit4_out[i].data, 0)
                    exit_counts[3] += 1
                    if predicted == labels[i]:
                        correct_exit4 += (predicted == labels[i]).item()

                elif entropy_exit5[i] < exit_thresholds[4]:
                    _, predicted = torch.max(exit5_out[i].data, 0)
                    exit_counts[4] += 1
                    if predicted == labels[i]:
                        correct_exit5 += (predicted == labels[i]).item()

                else:
                    _, predicted = torch.max(main_out[i].data, 0)
                    exit_counts[5] += 1
                    if predicted == labels[i]:
                        correct_main += (predicted == labels[i]).item()

                cnt += 1

    # Calculate accuracy and exit ratios
    exit1_accuracy = 100 * correct_exit1 / exit_counts[0] if exit_counts[0] > 0 else 0
    exit2_accuracy = 100 * correct_exit2 / exit_counts[1] if exit_counts[1] > 0 else 0
    exit3_accuracy = 100 * correct_exit3 / exit_counts[2] if exit_counts[2] > 0 else 0
    exit4_accuracy = 100 * correct_exit4 / exit_counts[3] if exit_counts[3] > 0 else 0
    exit5_accuracy = 100 * correct_exit5 / exit_counts[4] if exit_counts[4] > 0 else 0
    main_accuracy = 100 * correct_main / exit_counts[5] if exit_counts[5] > 0 else 0

    exit_ratios = [100 * count / total for count in exit_counts]

    return [exit1_accuracy, exit2_accuracy, exit3_accuracy, exit4_accuracy, exit5_accuracy, main_accuracy], exit_ratios

def threshold_inference_new(model, category, dataloader, exit_thresholds):
    model.eval()
    exit_point = []
    with torch.no_grad():

            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(dataloader)

            softmax_exit1 = F.softmax(exit1_out, dim=1)
            entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

            softmax_exit2 = F.softmax(exit2_out, dim=1)
            entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

            softmax_exit3 = F.softmax(exit3_out, dim=1)
            entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

            softmax_exit4 = F.softmax(exit4_out, dim=1)
            entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

            softmax_exit5 = F.softmax(exit5_out, dim=1)
            entropy_exit5 = -torch.sum(softmax_exit5 * torch.log(softmax_exit5 + 1e-5), dim=1)

            for i in range(dataloader.size(0)):
                if entropy_exit1[i] < exit_thresholds[0]:
                    _, predicted = torch.max(exit1_out.data, 1)
                    classified_label = predicted
                    exit_point.append(0)
                elif entropy_exit2[i] < exit_thresholds[1]:
                    _, predicted = torch.max(exit2_out.data, 1)
                    classified_label = predicted
                    exit_point.append(1)
                elif entropy_exit3[i] < exit_thresholds[2]:
                    _, predicted = torch.max(exit3_out.data, 1)
                    classified_label = predicted
                    exit_point.append(2)
                elif entropy_exit4[i] < exit_thresholds[3]:
                    _, predicted = torch.max(exit4_out.data, 1)
                    classified_label = predicted
                    exit_point.append(3)
                elif entropy_exit5[i] < exit_thresholds[4]:
                    _, predicted = torch.max(exit5_out.data, 1)
                    classified_label = predicted
                    exit_point.append(4)
                else:
                    _, predicted = torch.max(main_out.data, 1)
                    classified_label = predicted
                    exit_point.append(5)

    return classified_label, exit_point

def ee_inference():
    initial_thresholds = [1.0] * 5 # max entropy is 2.3 for 10 class problem, but starting at 1.0 is just faster

    initial_accuracy = simple_inference(model, testloader)

    print(f"simple_inference accuracy: {initial_accuracy}")

    # Dis/enable threshold finder
    FIND_THRESHOLD = False
    if FIND_THRESHOLD:
        # accuracy_bounds = [initial_accuracy[i] - 1 for i in range(5)] # allow 1% drop in accuracy
        optimal_thresholds, best_accuracies, best_exit_ratios = threshold_finder(model, trainloader, initial_thresholds,
                                                                             initial_accuracy, step=0.1,
                                                                             tolerance=10)
        print("---------------------------------")
        print(f"threshold_finder Thresholds: {optimal_thresholds}")
        print(f"threshold_finder Accuracies: {best_accuracies}")
        print(f"threshold_finder Exit Ratios: {best_exit_ratios}")
    else:
        optimal_thresholds = [0.5, 0.5, 0.7, 0.85, 0.5]
        print(f"optimal Thresholds already found: {optimal_thresholds}")

    # Do threshold inference, also store the wrong imgs locally
    accuracy, exit_ratios = threshold_inference(model, testloader, optimal_thresholds)

    print("---------------------------------")
    print(f"threshold_inference Accuracy: {accuracy}")
    print(f"threshold_inference Exit Ratios: {exit_ratios}")

def threshold_inference_with_class_stats(model, dataloader, exit_thresholds, num_classes=10):
    model.eval()

    # Track correct predictions per class and exit point
    class_correct = [[0 for _ in range(num_classes)] for _ in range(6)]  # 6 exits (5 early + main)
    class_total = [0 for _ in range(num_classes)]
    exit_counts = [0, 0, 0, 0, 0, 0]  # Count samples exiting at each point

    # Track exit distribution per class
    class_exit_distribution = [[0 for _ in range(6)] for _ in range(num_classes)]

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out, exit5_out = model(images)

            # Calculate softmax and entropy for all exits
            softmax_exit1 = F.softmax(exit1_out, dim=1)
            entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

            softmax_exit2 = F.softmax(exit2_out, dim=1)
            entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

            softmax_exit3 = F.softmax(exit3_out, dim=1)
            entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

            softmax_exit4 = F.softmax(exit4_out, dim=1)
            entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

            softmax_exit5 = F.softmax(exit5_out, dim=1)
            entropy_exit5 = -torch.sum(softmax_exit5 * torch.log(softmax_exit5 + 1e-5), dim=1)

            # Process each sample
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1

                # Determine exit point based on entropy thresholds
                if entropy_exit1[i] < exit_thresholds[0]:
                    _, predicted = torch.max(exit1_out[i].data, 0)
                    exit_point = 0
                    exit_counts[0] += 1
                    class_exit_distribution[label][0] += 1
                    if predicted == label:
                        class_correct[0][label] += 1

                elif entropy_exit2[i] < exit_thresholds[1]:
                    _, predicted = torch.max(exit2_out[i].data, 0)
                    exit_point = 1
                    exit_counts[1] += 1
                    class_exit_distribution[label][1] += 1
                    if predicted == label:
                        class_correct[1][label] += 1

                elif entropy_exit3[i] < exit_thresholds[2]:
                    _, predicted = torch.max(exit3_out[i].data, 0)
                    exit_point = 2
                    exit_counts[2] += 1
                    class_exit_distribution[label][2] += 1
                    if predicted == label:
                        class_correct[2][label] += 1

                elif entropy_exit4[i] < exit_thresholds[3]:
                    _, predicted = torch.max(exit4_out[i].data, 0)
                    exit_point = 3
                    exit_counts[3] += 1
                    class_exit_distribution[label][3] += 1
                    if predicted == label:
                        class_correct[3][label] += 1

                elif entropy_exit5[i] < exit_thresholds[4]:
                    _, predicted = torch.max(exit5_out[i].data, 0)
                    exit_point = 4
                    exit_counts[4] += 1
                    class_exit_distribution[label][4] += 1
                    if predicted == label:
                        class_correct[4][label] += 1

                else:
                    _, predicted = torch.max(main_out[i].data, 0)
                    exit_point = 5
                    exit_counts[5] += 1
                    class_exit_distribution[label][5] += 1
                    if predicted == label:
                        class_correct[5][label] += 1

    # Calculate and print results
    total_samples = sum(class_total)
    exit_ratios = [100 * count / total_samples for count in exit_counts]

    # Class names for CIFAR-10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    print("\n====== Threshold Inference Results ======")
    print(f"Exit Distribution: {exit_ratios}")
    print(f"Total samples processed: {total_samples}")

    # Print per-exit accuracy
    print("\n== Accuracy per Exit Point ==")
    for exit_idx in range(6):
        exit_name = f"Exit {exit_idx + 1}" if exit_idx < 5 else "Main Exit"
        if exit_counts[exit_idx] > 0:
            exit_correct = sum(class_correct[exit_idx])
            exit_acc = 100 * exit_correct / exit_counts[exit_idx]
            print(f"{exit_name}: {exit_acc:.2f}% ({exit_correct}/{exit_counts[exit_idx]})")
        else:
            print(f"{exit_name}: N/A (0 samples)")

    # Print per-class accuracy
    print("\n== Per-Class Accuracy ==")
    for class_idx in range(num_classes):
        if class_total[class_idx] > 0:
            class_correct_total = sum(class_correct[exit_idx][class_idx] for exit_idx in range(6))
            class_acc = 100 * class_correct_total / class_total[class_idx]
            print(f"{classes[class_idx]}: {class_acc:.2f}% ({class_correct_total}/{class_total[class_idx]})")

    # Print exit distribution per class
    print("\n== Exit Distribution per Class ==")
    for class_idx in range(num_classes):
        if class_total[class_idx] > 0:
            dist = [100 * class_exit_distribution[class_idx][exit_idx] / class_total[class_idx]
                    for exit_idx in range(6)]
            print(f"{classes[class_idx]}: Exit1: {dist[0]:.1f}%, Exit2: {dist[1]:.1f}%, "
                  f"Exit3: {dist[2]:.1f}%, Exit4: {dist[3]:.1f}%, Exit5: {dist[4]:.1f}%, Main: {dist[5]:.1f}%")

    return exit_ratios, class_correct, class_total, class_exit_distribution

def white_board_test(image_path):
    # Load and preprocess a single image for testing
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # Define the same transforms used for the test dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()  # Keep a copy for display
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Optimal thresholds for the BranchedResNet50
    optimal_thresholds = [0.5, 0.5, 0.7, 0.85, 0.5]

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Get the prediction and exit point using threshold_inference_new
        classified_label, exit_point = threshold_inference_new(model, None, image_tensor, optimal_thresholds)

    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title(f'Resnet50 EE pattern on VGG16 \n'
              f'label: {classified_label.item()} , Exit: {exit_point[0]}')
    plt.show()

if __name__ == '__main__':

    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=100, num_workers=2)

    # ee_inference()

    white_board_test("Results/white_board/resnet50/0_airplane/white_board_20250403_171932.png")
    white_board_test("Results/white_board/resnet50/2_bird/white_board_20250403_200709.png")
    white_board_test("Results/white_board/resnet50/4_deer/white_board_20250404_001914.png")
    white_board_test("Results/white_board/resnet50/6_frog/white_board_20250404_120430.png")

    # branch classifier results
    # optimal_thresholds = [0.5, 0.5, 0.7, 0.85, 0.5]
    # threshold_inference_with_class_stats(model, testloader, optimal_thresholds)


# simple_inference accuracy: [58.5, 69.51, 78.93, 90.31, 95.48, 95.8]
# optimal Thresholds already found: [0.5, 0.5, 0.7, 0.85, 0.5]
# ---------------------------------
# threshold_inference Accuracy: [92.16090768437338, 92.83299526707235, 92.2420796100731, 93.26747164288328, 93.73695198329854, 62.23776223776224]
# threshold_inference Exit Ratios: [19.39, 14.79, 24.62, 27.33, 9.58, 4.29]
