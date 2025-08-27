import os
import zipfile
import re
import logging
from collections import defaultdict


def compress_folder(folder_path, output_zip):
    """
    Compresses the contents of a folder into a zip file, including all subdirectories.

    :param folder_path: Path to the folder to compress.
    :param output_zip: Path for the resulting zip file.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return False

    # Normalize paths
    folder_path = os.path.abspath(folder_path)
    output_zip = os.path.abspath(output_zip)

    try:
        # Create the zip file
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Track counts for reporting
            file_count = 0
            dir_count = 0

            # Walk through the directory structure
            for root, dirs, files in os.walk(folder_path):
                # Add directories (empty directories will be included)
                for dirname in dirs:
                    dir_path = os.path.join(root, dirname)
                    relative_path = os.path.relpath(dir_path, folder_path)
                    # Some zip utilities treat directories specially by ending with a slash
                    zipf.write(dir_path, relative_path + '/')
                    dir_count += 1

                # Add files
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, folder_path)
                    zipf.write(full_path, relative_path)
                    file_count += 1

        print(f"Successfully compressed {file_count} files and {dir_count} directories into '{output_zip}'")
        return True

    except Exception as e:
        print(f"Error compressing folder: {e}")
        return False


def get_files_remove_png():
    # Directory to scan
    directory_path = "Results/white_board/CIFAR100/Resnet50"

    # Set up logging - simple format with just the file names
    logging.basicConfig(
        filename='cleaned_filenames.log',
        level=logging.INFO,
        format='%(message)s'
    )

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    try:
        # Get all files in the directory
        all_files = os.listdir(directory_path)

        # Process each file
        for filename in all_files:
            # Remove .png extension from the filename
            cleaned_name = filename.replace('.png', '')

            # Write the cleaned name to the log file
            logging.info(f"{cleaned_name}")

        print(f"Processed {len(all_files)} files")
        print(f"Cleaned filenames (without .png) saved to 'cleaned_filenames.log'")

    except Exception as e:
        print(f"Error processing files: {e}")

def group_files_by_losspara():
    # Dictionary to store grouped filenames
    grouped_files = defaultdict(list)

    # Read the log file
    log_file = 'cleaned_filenames.log'

    try:
        with open(log_file, 'r') as file:
            for line in file:
                line = line.strip()

                # Extract LossPara value using regex
                match = re.match(r'LossPara_(\d+)_', line)
                if match:
                    losspara_value = match.group(1)
                    grouped_files[losspara_value].append(line)

        # Create output file with grouped information
        with open('grouped_by_losspara.log', 'w') as output:
            # Sort the dictionary by LossPara value
            for losspara in sorted(grouped_files.keys(), key=int):
                output.write(f"LossPara_{losspara} ({len(grouped_files[losspara])} files):\n")
                for filename in sorted(grouped_files[losspara]):
                    output.write(f"  {filename}\n")
                output.write("\n")

        print(f"Files grouped by LossPara value in 'grouped_by_losspara.log'")
        print(f"Found {len(grouped_files)} different LossPara values")
        total_files = sum(len(files) for files in grouped_files.values())
        print(f"Total of {total_files} files processed")

    except Exception as e:
        print(f"Error processing log file: {e}")


import re


def count_matching_labels(log_file_path):
    # Read the log file
    with open(log_file_path, 'r') as f:
        log_text = f.read()

    # Define pattern to extract Label and GenLabel numbers
    pattern = r"Label_(\d+):GenLabel_(\d+)_"
    matches = re.findall(pattern, log_text)

    # Count matches
    total_entries = len(matches)
    matching_pairs = sum(1 for label, genlabel in matches if label == genlabel)

    # Group by LossPara
    groups = {}
    for line in log_text.split('\n'):
        if 'LossPara_' in line and 'Label_' in line and 'GenLabel_' in line:
            loss_para = re.search(r'LossPara_(\d+)', line).group(1)
            label = re.search(r'Label_(\d+):', line).group(1)
            genlabel = re.search(r'GenLabel_(\d+)_', line).group(1)

            if loss_para not in groups:
                groups[loss_para] = {'total': 0, 'matches': 0}
            groups[loss_para]['total'] += 1
            if label == genlabel:
                groups[loss_para]['matches'] += 1

    # Print results
    print("\nBreakdown by LossPara:")
    for loss_para, stats in sorted(groups.items(), key=lambda x: int(x[0])):
        print(
            f"LossPara_{loss_para}: {stats['matches']} matches out of {stats['total']} ({stats['matches'] / stats['total'] * 100:.1f}%)")

    return matching_pairs, total_entries, groups



if __name__ == "__main__":

    # # Example Usage
    folder_to_compress = "/home/yibo/PycharmProjects/Thesis/Results/white_board/transfer test/ViT"  # Replace with your folder path
    output_zip_file = "1.zip"  # Replace with desired output file name or path
    compress_folder(folder_to_compress, output_zip_file)

    # folder_to_compress = "Results/Sweep Paras/CIFAR100_B_Resnet50"  # Replace with your folder path
    # output_zip_file = "Sweep_CIFAR100_Res50.zip"  # Replace with desired output file name or path
    # compress_folder(folder_to_compress, output_zip_file)



    # get_files_remove_png()
    # group_files_by_losspara()
    # matches, total, groups = count_matching_labels('grouped_by_losspara.log')

