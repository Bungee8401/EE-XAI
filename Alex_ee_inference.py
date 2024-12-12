import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import alexnet
import matplotlib.pyplot as plt
from Alexnet_early_exit import BranchedAlexNet
from torch.utils.tensorboard import SummaryWriter

# seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)

# Load the trained model
model = BranchedAlexNet(num_classes=10).to(device)
model.load_state_dict(torch.load(r"D:\Study\Module\Master Thesis\trained_models\B-Alex lr=0.001 transfer learning\B-Alex_cifar10_epoch_30.pth",
                                 weights_only=True))

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
    upper_acc = accuracy[-1] #91.7% for this checkpoint

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

    return [exit1_accuracy, exit2_accuracy, exit3_accuracy, exit4_accuracy, exit5_accuracy, main_accuracy], exit_ratios

def _exit_point_datasets(model, dataloader, exit_thresholds, base_dir): # TODO: need test
    import os
    import torch.utils.data

    model.eval()
    exit_datasets = [[] for _ in range(6)]  # 6 datasets for 5 exits and main

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)

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

            # Determine exit points based on thresholds
            for i in range(labels.size(0)):
                if entropy_exit1[i] < exit_thresholds[0]:
                    exit_datasets[0].append((images[i].cpu(), labels[i].cpu()))
                elif entropy_exit2[i] < exit_thresholds[1]:
                    exit_datasets[1].append((images[i].cpu(), labels[i].cpu()))
                elif entropy_exit3[i] < exit_thresholds[2]:
                    exit_datasets[2].append((images[i].cpu(), labels[i].cpu()))
                elif entropy_exit4[i] < exit_thresholds[3]:
                    exit_datasets[3].append((images[i].cpu(), labels[i].cpu()))
                elif entropy_exit5[i] < exit_thresholds[4]:
                    exit_datasets[4].append((images[i].cpu(), labels[i].cpu()))
                else:
                    exit_datasets[5].append((images[i].cpu(), labels[i].cpu()))

    # Convert lists to TensorDataset
    exit_datasets = [
        torch.utils.data.TensorDataset(torch.stack([x[0] for x in dataset]), torch.stack([x[1] for x in dataset])) for dataset in exit_datasets]

    # Save datasets locally
    os.makedirs(base_dir, exist_ok=True)
    for idx, dataset in enumerate(exit_datasets):
        torch.save(dataset, os.path.join(base_dir, f'exit_dataset_{idx}.pt'))

    print("_exit_point_datasets ready!")
    pass

if __name__ == '__main__':

    # Initialize TensorBoard
    log_dir = 'Tensorboard_data/Alex_ee_inference'
    writer = SummaryWriter(log_dir=log_dir)

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='D:/Study/Module/Master Thesis/dataset/CIFAR10', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    initial_thresholds = [1.0] * 5 # max entropy is 2.3 for 10 class problem, but starting at 1.0 is just faster

    initial_accuracy = simple_inference(model, testloader)

    print(f"simple_inference accuracy: {initial_accuracy}")

    # accuracy_bounds = [initial_accuracy[i] - 1 for i in range(5)] # allow 1% drop in accuracy
    optimal_thresholds, best_accuracies, best_exit_ratios = threshold_finder(model, testloader, initial_thresholds,
                                                                             initial_accuracy, step=0.1,
                                                                             tolerance=10)
    print("---------------------------------")
    print(f"threshold_finder Thresholds: {optimal_thresholds}")
    print(f"threshold_finder Accuracies: {best_accuracies}")
    print(f"threshold_finder Exit Ratios: {best_exit_ratios}")

    accuracy, exit_ratios = threshold_inference(model, testloader, optimal_thresholds)

    print("---------------------------------")
    print(f"threshold_inference Accuracy: {accuracy}")
    print(f"threshold_inference Exit Ratios: {exit_ratios}")

    # Log variables to TensorBoard
    writer.add_scalars('Accuracy', {
        'Exit1': accuracy[0],
        'Exit2': accuracy[1],
        'Exit3': accuracy[2],
        'Exit4': accuracy[3],
        'Exit5': accuracy[4],
        'Main': accuracy[5]
    })

    writer.add_scalars('Exit Ratios', {
        'Exit1': exit_ratios[0],
        'Exit2': exit_ratios[1],
        'Exit3': exit_ratios[2],
        'Exit4': exit_ratios[3],
        'Exit5': exit_ratios[4],
        'Main': exit_ratios[5]
    })

    # Close the TensorBoard writer
    writer.close()

#TODO： 1. threshold_inference function, uses the entropy as the criteria; could be better methods -> paper EENet
#       2. threshold_finder function, finding thresholds sequentially; wonder if better
#       3. threshold_finder function, tolerance dont need i think

# simple_inference accuracy: [63.23, 79.73, 86.3, 89.65, 91.71, 91.79]
# ----finding optimal threshold for early exit point---- 1
# current thresh: 0.9, Accuracy: 80.96427199385325, Exit Ratio: 52.06
# current thresh: 0.8, Accuracy: 83.19381841596909, Exit Ratio: 46.59
# current thresh: 0.7000000000000001, Accuracy: 85.39461020211742, Exit Ratio: 41.56
# current thresh: 0.6000000000000001, Accuracy: 87.78416872089838, Exit Ratio: 36.51
# current thresh: 0.5000000000000001, Accuracy: 90.03513254551261, Exit Ratio: 31.31
# current thresh: 0.40000000000000013, Accuracy: 92.07660533233195, Exit Ratio: 26.63
# Optimal thresh: 0.5000000000000001, Accuracy: 92.07660533233195, Exit Ratio: 26.63
# ----finding optimal threshold for early exit point---- 2
# current thresh: 0.9, Accuracy: 84.05034754837497, Exit Ratio: 53.23
# current thresh: 0.8, Accuracy: 85.51458670988654, Exit Ratio: 49.36
# current thresh: 0.7000000000000001, Accuracy: 87.51671868033883, Exit Ratio: 44.86
# current thresh: 0.6000000000000001, Accuracy: 89.775, Exit Ratio: 40.0
# current thresh: 0.5000000000000001, Accuracy: 92.1351504826803, Exit Ratio: 35.22
# Optimal thresh: 0.6000000000000001, Accuracy: 92.1351504826803, Exit Ratio: 35.22
# ----finding optimal threshold for early exit point---- 3
# current thresh: 0.9, Accuracy: 78.3498759305211, Exit Ratio: 32.24
# current thresh: 0.8, Accuracy: 80.2233902759527, Exit Ratio: 30.44
# current thresh: 0.7000000000000001, Accuracy: 82.01058201058201, Exit Ratio: 28.35
# current thresh: 0.6000000000000001, Accuracy: 84.47254049782694, Exit Ratio: 25.31
# current thresh: 0.5000000000000001, Accuracy: 87.01707097933513, Exit Ratio: 22.26
# current thresh: 0.40000000000000013, Accuracy: 88.9795918367347, Exit Ratio: 19.6
# current thresh: 0.30000000000000016, Accuracy: 90.40047114252062, Exit Ratio: 16.98
# current thresh: 0.20000000000000015, Accuracy: 92.10526315789474, Exit Ratio: 14.06
# Optimal thresh: 0.30000000000000016, Accuracy: 92.10526315789474, Exit Ratio: 14.06
# ----finding optimal threshold for early exit point---- 4
# current thresh: 0.9, Accuracy: 76.41597028783659, Exit Ratio: 21.54
# current thresh: 0.8, Accuracy: 77.59803921568627, Exit Ratio: 20.4
# current thresh: 0.7000000000000001, Accuracy: 79.07098121085595, Exit Ratio: 19.16
# current thresh: 0.6000000000000001, Accuracy: 81.18527042577675, Exit Ratio: 17.38
# current thresh: 0.5000000000000001, Accuracy: 83.15926892950391, Exit Ratio: 15.32
# current thresh: 0.40000000000000013, Accuracy: 86.34001484780995, Exit Ratio: 13.47
# current thresh: 0.30000000000000016, Accuracy: 87.54237288135593, Exit Ratio: 11.8
# current thresh: 0.20000000000000015, Accuracy: 89.12175648702595, Exit Ratio: 10.02
# current thresh: 0.10000000000000014, Accuracy: 91.92307692307692, Exit Ratio: 7.8
# Optimal thresh: 0.20000000000000015, Accuracy: 91.92307692307692, Exit Ratio: 7.8
# ----finding optimal threshold for early exit point---- 5
# current thresh: 0.9, Accuracy: 77.15487035739314, Exit Ratio: 14.27
# current thresh: 0.8, Accuracy: 78.51851851851852, Exit Ratio: 13.5
# current thresh: 0.7000000000000001, Accuracy: 79.76190476190476, Exit Ratio: 12.6
# current thresh: 0.6000000000000001, Accuracy: 82.57839721254355, Exit Ratio: 11.48
# current thresh: 0.5000000000000001, Accuracy: 84.7609561752988, Exit Ratio: 10.04
# current thresh: 0.40000000000000013, Accuracy: 87.31596828992072, Exit Ratio: 8.83
# current thresh: 0.30000000000000016, Accuracy: 89.20676202860858, Exit Ratio: 7.69
# current thresh: 0.20000000000000015, Accuracy: 90.89529590288316, Exit Ratio: 6.59
# current thresh: 0.10000000000000014, Accuracy: 93.13725490196079, Exit Ratio: 5.1
# Optimal thresh: 0.20000000000000015, Accuracy: 93.13725490196079, Exit Ratio: 5.1
# ---------------------------------
# threshold_finder Thresholds: [0.5000000000000001, 0.6000000000000001, 0.30000000000000016, 0.20000000000000015, 0.20000000000000015]
# threshold_finder Accuracies: [92.07660533233195, 92.1351504826803, 92.10526315789474, 91.92307692307692, 93.13725490196079]
# threshold_finder Exit Ratios: [26.63, 35.22, 14.06, 7.8, 5.1]
# ---------------------------------
# threshold_inference Accuracy: [92.07660533233195, 92.1351504826803, 92.10526315789474, 91.92307692307692, 93.13725490196079, 62.734584450402146]
# threshold_inference Exit Ratios: [26.63, 35.22, 14.06, 7.8, 5.1, 11.19]

# TODO: other metrics may need to be added to optimize the threshold?
#           - if no, can continue to test what kind of changes have biggest impact on acc; flops, params, latency are not the concern here
#           - so, I need three functions:
#               + input_transform function
#                   - have (testloader, ect) as input,
#                   - (testloader_transformed) as output
#               + acc_and_exit_ratio function
#                   - (model, testloader, testloader_transformed) as input
#                   - (acc, acc_trans, exit_ratio, exit_ratio_trans) as output
#               + data_split function
#                   - split data according to class
#                   - split easy and hard
#                   - split easy and hard according to class

def image_masking(images, pattern, location): # TODO: try to make locations generated randomly
    masked_images = images.clone()
    for loc in location:
        x, y = loc
        if pattern == 'checkerboard':
            for i in range(x, x + 8):
                for j in range(y, y + 8):
                    if (i + j) % 2 == 0:
                        masked_images[:, :, i, j] = 0
        elif pattern == 'stripe':
            for i in range(x, x + 8):
                masked_images[:, :, i, y:y + 8] = 0
        elif pattern == 'cropping':
            for i in range(x, x + 8):
                for j in range(y, y + 8):
                    masked_images[:, :, 32:32-i, 32:32-j] = 0
        # Add more patterns as needed
    return masked_images

def input_transform(testloader, transform_type):
    transformed_data = []

    for data in testloader:
        images, labels = data
        if transform_type == 'color':
            # Set R, G, B channels to 0
            images[:, 0, :, :] = 0  # Set Red channel to 0
            images[:, 1, :, :] = 0  # Set Green channel to 0
            images[:, 2, :, :] = 0  # Set Blue channel to 0
        elif transform_type == 'masking':
            # Apply some masking transformation
            # mask = torch.ones_like(images)
            # mask[:, :, 112:, 112:] = 0  # Example mask
            # images = images * mask
            images = image_masking(images, "xxx_TYPE_xxx")
        # Add more transformations

        transformed_data.append((images, labels))

    testloader_transformed = torch.utils.data.DataLoader(transformed_data, batch_size=testloader.batch_size, shuffle=False, num_workers=testloader.num_workers)

    return testloader_transformed

def acc_and_exit_ratio(model, testloader, testloader_transformed):
    accuracy, exit_ratios = threshold_inference(model, testloader, [1.0] * 5)
    accuracy_trans, exit_ratios_trans = threshold_inference(model, testloader_transformed, [1.0] * 5)

    return accuracy, accuracy_trans, exit_ratios, exit_ratios_trans

def data_split(dataset, split_type='random'):
    class_data = {}
    for data in dataset:
        images, labels = data
        for i in range(len(labels)):
            label = labels[i].item()
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((images[i], label))

    easy_data = []
    hard_data = []

    if split_type == 'half':
        for label, data in class_data.items():
            split_idx = len(data) // 2
            easy_data.extend(data[:split_idx])
            hard_data.extend(data[split_idx:])
        return
    elif split_type == 'exit_location':
        exit_location_path = _exit_point_datasets(BranchedAlexNet, testloader, [0.2]*5)
        exit_datasets = [torch.load(os.path.join(exit_location_path, f'exit_dataset_{i}.pt')) for i in range(6)]
        return exit_datasets
    elif split_type == 'easy_hard':
        for label, data in class_data.items():
            split_idx = len(data) // 2
            easy_data.extend(data[:split_idx])
            hard_data.extend(data[split_idx:])
        return

    pass