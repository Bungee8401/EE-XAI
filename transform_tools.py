import torch
import os
import shutil

# TODO: I now have two data set folders
#      1, determine which class I want to look at first
#      2, use early correct samples to help later correct samples
#      3, use correct samples to help wrong samples

#TODO 0: transformations on which data?
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
    elif split_type == 'easy_hard':
        for label, data in class_data.items():
            split_idx = len(data) // 2
            easy_data.extend(data[:split_idx])
            hard_data.extend(data[split_idx:])
        return

    pass

def one_class_set(src_folder, dest_folder, class_name):
    for root, dirs, files in os.walk(src_folder):
        for filename in files:
            if filename.startswith(class_name):
                src_path = os.path.join(root, filename)
                relative_path = os.path.relpath(src_path, src_folder)
                dest_path = os.path.join(dest_folder, relative_path)
                dest_dir = os.path.dirname(dest_path)

                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                shutil.copy(src_path, dest_dir)

#TODO 1: transformation
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


#TODO 2: transformations influence on accuracy and exit ratio

# def acc_and_exit_ratio(model, testloader, testloader_transformed):
#     accuracy, exit_ratios = threshold_inference(model, testloader, [1.0] * 5)
#     accuracy_trans, exit_ratios_trans = threshold_inference(model, testloader_transformed, [1.0] * 5)
#
#     return accuracy, accuracy_trans, exit_ratios, exit_ratios_trans



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


if __name__ == '__main__':

    ONE_CLASS = False
    if ONE_CLASS:
        one_class_set('Alex_thresh_wrong', 'Airplane_wrong', 'airplane')
        one_class_set('Alex_thresh_right', 'Airplane_right', 'airplane')
    else
        pass

