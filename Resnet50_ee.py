import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.models import resnet50
import time
from CustomDataset import Data_prep_224_normal_N

class BranchedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(BranchedResNet50, self).__init__()
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

        # Branch 1 after layer1
        self.branch1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56*56*256, num_classes)
        )

        # Branch 2 after layer2
        self.branch2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*512, num_classes)
        )

        # Branch 3 after layer3
        self.branch3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14*14*1024, num_classes)
        )

        # Branch 4 after layer4
        self.branch4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out_branch1 = self.branch1(x)

        x = self.layer2(x)
        out_branch2 = self.branch2(x)

        x = self.layer3(x)
        out_branch3 = self.branch3(x)

        x = self.layer4(x)
        out_branch4 = self.branch4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out_main = self.fc(x)

        return out_main, out_branch1, out_branch2, out_branch3, out_branch4

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features_layer1 = self.layer1(x)
        features_layer2 = self.layer2(features_layer1)
        features_layer3 = self.layer3(features_layer2)
        features_layer4 = self.layer4(features_layer3)

        x = self.avgpool(features_layer4)
        x = torch.flatten(x, 1)

        return x

def initialize_model():

    pytorch_lightning.seed_everything(2024)
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = resnet50(weights='IMAGENET1K_V1')

    # Initialize the BranchedResNet50 model
    model = BranchedResNet50(num_classes=10).to(device)

    # Copy the weights from the pretrained ResNet50 model to the BranchedResNet50 model
    model.conv1.load_state_dict(resnet.conv1.state_dict())
    model.bn1.load_state_dict(resnet.bn1.state_dict())
    model.layer1.load_state_dict(resnet.layer1.state_dict())
    model.layer2.load_state_dict(resnet.layer2.state_dict())
    model.layer3.load_state_dict(resnet.layer3.state_dict())
    model.layer4.load_state_dict(resnet.layer4.state_dict())
    model.avgpool.load_state_dict(resnet.avgpool.state_dict())

    root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
    dataprep = Data_prep_224_normal_N(root)
    trainloader, valloader, testloader = dataprep.create_loaders(batch_size=128, num_workers=2)

    return model, device, trainloader, valloader, testloader

def train(num_epochs):
    # Lists to store train and validation losses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_exit1 = []
    val_exit2 = []
    val_exit3 = []
    val_exit4 = []

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time for the epoch
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            main_out, exit1_out, exit2_out, exit3_out, exit4_out = model(inputs)

            # Calculate individual losses
            loss_exit1 = criterion(exit1_out, labels)
            loss_exit2 = criterion(exit2_out, labels)
            loss_exit3 = criterion(exit3_out, labels)
            loss_exit4 = criterion(exit4_out, labels)
            loss_main = criterion(main_out, labels)

            # Combine losses (average or weighted sum)
            # loss_all = (loss_exit1 + loss_exit2 + loss_main) / 3
            loss_all = (  (1/5)*loss_exit1
                        + (1/4)*loss_exit2
                        + (1/3)*loss_exit3
                        + (1/2)*loss_exit4
                        + loss_main)            #to alleviate gradient imbalance issue; paper EECE

            loss_all.backward()
            optimizer.step()

            running_loss += loss_all.item()

            # Calculate training accuracy for this batch
            _, predicted = torch.max(main_out.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        train_accuracies.append(100 * correct_train / total_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        correct_1 = 0
        correct_2 = 0
        correct_3 = 0
        correct_4 = 0
        correct_5 = 0
        total_val = 0

        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                main_out, exit1_out, exit2_out, exit3_out, exit4_out = model(images)

                # Calculate individual losses
                loss_exit1 = criterion(exit1_out, labels)
                loss_exit2 = criterion(exit2_out, labels)
                loss_exit3 = criterion(exit3_out, labels)
                loss_exit4 = criterion(exit4_out, labels)
                loss_main = criterion(main_out, labels)

                # Combine losses (average them for now)
                # loss_all = (loss_exit1 + loss_exit2 + loss_main) / 3
                loss_all = ((1 / 5) * loss_exit1
                            + (1 / 4) * loss_exit2
                            + (1 / 3) * loss_exit3
                            + (1 / 2) * loss_exit4
                            + loss_main)  # to alleviate gradient imbalance issue; paper EECE

                val_loss += loss_all.item()

                _, predicted_main = torch.max(main_out.data, 1)
                _, predicted_exit1 = torch.max(exit1_out.data, 1)
                _, predicted_exit2 = torch.max(exit2_out.data, 1)
                _, predicted_exit3 = torch.max(exit3_out.data, 1)
                _, predicted_exit4 = torch.max(exit4_out.data, 1)

                total_val += labels.size(0)

                correct_val += (predicted_main == labels).sum().item()
                correct_1 += (predicted_exit1 == labels).sum().item()
                correct_2 += (predicted_exit2 == labels).sum().item()
                correct_3 += (predicted_exit3 == labels).sum().item()
                correct_4 += (predicted_exit4 == labels).sum().item()

        scheduler.step()

        val_loss /= len(valloader)
        val_losses.append(val_loss)

        val_accuracies.append(100 * correct_val / total_val)
        val_exit1.append(100 * correct_1 / total_val)
        val_exit2.append(100 * correct_2 / total_val)
        val_exit3.append(100 * correct_3 / total_val)
        val_exit4.append(100 * correct_4 / total_val)


        epoch_end_time = time.time()  # End time for the epoch
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        epoch_hours, epoch_remainder = divmod(epoch_elapsed_time, 3600)
        epoch_minutes, epoch_seconds = divmod(epoch_remainder, 60)

        print(f"Epoch [{epoch + 1}/{num_epochs}], avg Train Loss: {epoch_loss:.4f}, avg Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], main Train Accuracy: {train_accuracies[-1]:.2f}%, main Val Accuracy: {val_accuracies[-1]:.2f}%")
        print(f"Exit 1 val Accuracy: {val_exit1[-1]:.2f}%, Exit 2 val Accuracy: {val_exit2[-1]:.2f}%, Exit 3 val Accuracy: {val_exit3[-1]:.2f}%, Exit 4 val Accuracy: {val_exit4[-1]:.2f}%")
        print(f"Time per epoch: {int(epoch_hours)}h {int(epoch_minutes)}m {int(epoch_seconds)}s")


        #save every 10 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f'/home/yibo/PycharmProjects/Thesis/training_weights/resnet50/B-Resnet50_epoch_{epoch + 1}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    # Save the trained model
    torch.save(model.state_dict(), '/home/yibo/PycharmProjects/Thesis/training_weights/resnet50/B-Resnet50_cifar10.pth')

    # Plotting the losses
    epoch = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_losses, label='Train Loss')
    plt.plot(epoch, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch, train_accuracies, label='Train Accuracy')
    plt.plot(epoch, val_accuracies, label='Validation Accuracy')
    plt.plot(epoch, val_exit1, label='Exit 1 Accuracy')
    plt.plot(epoch, val_exit2, label='Exit 2 Accuracy')
    plt.plot(epoch, val_exit3, label='Exit 3 Accuracy')
    plt.plot(epoch, val_exit4, label='Exit 4 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()

    # play_sound()
    plt.show()
    print('Finished Training')

def test():
    # Test the model after training
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    correct_5 = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            main_out, exit1_out, exit2_out, exit3_out, exit4_out = model(images)

            # Calculate individual losses
            loss_exit1 = criterion(exit1_out, labels)
            loss_exit2 = criterion(exit2_out, labels)
            loss_exit3 = criterion(exit3_out, labels)
            loss_exit4 = criterion(exit4_out, labels)

            loss_main = criterion(main_out, labels)

            # Combine losses (average them for now)
            loss_all = ((1 / 5) * loss_exit1
                        + (1 / 4) * loss_exit2
                        + (1 / 3) * loss_exit3
                        + (1 / 2) * loss_exit4
                        + loss_main)  # to alleviate gradient imbalance issue; paper EECE

            test_loss += loss_all.item()

            _, predicted_main = torch.max(main_out.data, 1)
            _, predicted_exit1 = torch.max(exit1_out.data, 1)
            _, predicted_exit2 = torch.max(exit2_out.data, 1)
            _, predicted_exit3 = torch.max(exit3_out.data, 1)
            _, predicted_exit4 = torch.max(exit4_out.data, 1)

            total_test += labels.size(0)
            correct_test += (predicted_main == labels).sum().item()
            correct_1 += (predicted_exit1 == labels).sum().item()
            correct_2 += (predicted_exit2 == labels).sum().item()
            correct_3 += (predicted_exit3 == labels).sum().item()
            correct_4 += (predicted_exit4 == labels).sum().item()


    test_loss /= len(testloader)
    test_accuracy = 100 * correct_test / total_test
    exit1_accuracy = 100 * correct_1 / total_test
    exit2_accuracy = 100 * correct_2 / total_test
    exit3_accuracy = 100 * correct_3 / total_test
    exit4_accuracy = 100 * correct_4 / total_test


    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
          f"Exit 1 Accuracy: {exit1_accuracy:.2f}%, Exit 2 Accuracy: {exit2_accuracy:.2f}%, "
          f"Exit 3 Accuracy: {exit3_accuracy:.2f}%, Exit 4 Accuracy: {exit4_accuracy:.2f}%  ")

def simple_inference(model, dataloader):
    model.eval()
    correct_main = 0
    correct_exit1 = 0
    correct_exit2 = 0
    correct_exit3 = 0
    correct_exit4 = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out = model(images)

            # Calculate predictions for each exit
            _, predicted_main = torch.max(main_out.data, 1)
            _, predicted_exit1 = torch.max(exit1_out.data, 1)
            _, predicted_exit2 = torch.max(exit2_out.data, 1)
            _, predicted_exit3 = torch.max(exit3_out.data, 1)
            _, predicted_exit4 = torch.max(exit4_out.data, 1)

            correct_main += (predicted_main == labels).sum().item()
            correct_exit1 += (predicted_exit1 == labels).sum().item()
            correct_exit2 += (predicted_exit2 == labels).sum().item()
            correct_exit3 += (predicted_exit3 == labels).sum().item()
            correct_exit4 += (predicted_exit4 == labels).sum().item()

    accuracy_main = 100 * correct_main / total
    accuracy_exit1 = 100 * correct_exit1 / total
    accuracy_exit2 = 100 * correct_exit2 / total
    accuracy_exit3 = 100 * correct_exit3 / total
    accuracy_exit4 = 100 * correct_exit4 / total

    return [accuracy_exit1, accuracy_exit2, accuracy_exit3, accuracy_exit4, accuracy_main]

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

    total = 0
    exit_counts = [0, 0, 0, 0, 0]
    cnt = 0

    # entropy-based criteria as in BranchyNet; in cifar10, max entropy is 2.3; smaller entropy, more confident
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            total += labels.size(0)

            # Forward pass
            main_out, exit1_out, exit2_out, exit3_out, exit4_out = model(images)

            # Calculate softmax and entropy for exit1-5
            softmax_exit1 = F.softmax(exit1_out, dim=1)
            entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

            softmax_exit2 = F.softmax(exit2_out, dim=1)
            entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

            softmax_exit3 = F.softmax(exit3_out, dim=1)
            entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

            softmax_exit4 = F.softmax(exit4_out, dim=1)
            entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

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

                else:
                    _, predicted = torch.max(main_out[i].data, 0)
                    exit_counts[4] += 1
                    if predicted == labels[i]:
                        correct_main += (predicted == labels[i]).item()

                cnt += 1

    # Calculate accuracy and exit ratios
    exit1_accuracy = 100 * correct_exit1 / exit_counts[0] if exit_counts[0] > 0 else 0
    exit2_accuracy = 100 * correct_exit2 / exit_counts[1] if exit_counts[1] > 0 else 0
    exit3_accuracy = 100 * correct_exit3 / exit_counts[2] if exit_counts[2] > 0 else 0
    exit4_accuracy = 100 * correct_exit4 / exit_counts[3] if exit_counts[3] > 0 else 0
    main_accuracy = 100 * correct_main / exit_counts[4] if exit_counts[4] > 0 else 0

    exit_ratios = [100 * count / total for count in exit_counts]

    return [exit1_accuracy, exit2_accuracy, exit3_accuracy, exit4_accuracy, main_accuracy], exit_ratios

def threshold_inference_new(model, category, dataloader, exit_thresholds):
    model.eval()
    exit_point = []
    with torch.no_grad():

            main_out, exit1_out, exit2_out, exit3_out, exit4_out = model(dataloader)

            softmax_exit1 = F.softmax(exit1_out, dim=1)
            entropy_exit1 = -torch.sum(softmax_exit1 * torch.log(softmax_exit1 + 1e-5), dim=1)

            softmax_exit2 = F.softmax(exit2_out, dim=1)
            entropy_exit2 = -torch.sum(softmax_exit2 * torch.log(softmax_exit2 + 1e-5), dim=1)

            softmax_exit3 = F.softmax(exit3_out, dim=1)
            entropy_exit3 = -torch.sum(softmax_exit3 * torch.log(softmax_exit3 + 1e-5), dim=1)

            softmax_exit4 = F.softmax(exit4_out, dim=1)
            entropy_exit4 = -torch.sum(softmax_exit4 * torch.log(softmax_exit4 + 1e-5), dim=1)

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
                else:
                    _, predicted = torch.max(main_out.data, 1)
                    classified_label = predicted
                    exit_point.append(5)

    return classified_label, exit_point

def ee_inference():
    initial_thresholds = [1.0] * 5  # max entropy is 2.3 for 10 class problem, but starting at 1.0 is just faster

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
        optimal_thresholds = [0.3, 0.45, 0.7, 0.05]
        print(f"optimal Thresholds already found: {optimal_thresholds}")

    # Do threshold inference, also store the wrong imgs locally
    accuracy, exit_ratios = threshold_inference(model, testloader, optimal_thresholds)

    print("---------------------------------")
    print(f"threshold_inference Accuracy: {accuracy}")
    print(f"threshold_inference Exit Ratios: {exit_ratios}")

if __name__ == '__main__':

    pl.seed_everything(2024)
    model, device, trainloader, valloader, testloader = initialize_model()

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    TRAIN = False
    if TRAIN:
        train(50)
        test()
    else:
        model.load_state_dict(torch.load(r"weights/Resnet50/B-Resnet50_epoch_10.pth", weights_only=True))
        test()

    ee_inference()


# simple_inference accuracy: [67.39, 76.58, 92.53, 96.4, 96.5]
# optimal Thresholds already found: [0.3, 0.45, 0.7, 0.05]
# ---------------------------------
# threshold_inference Accuracy: [96.51785714285714, 96.8141592920354, 96.57829141173805, 96.84106614017769, 71.83098591549296]
# threshold_inference Exit Ratios: [22.4, 16.95, 44.13, 10.13, 6.39]