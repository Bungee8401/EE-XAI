import os
import zipfile


def compress_folder(folder_path, output_zip):
    """
    Compresses the contents of a folder into a zip file.

    :param folder_path: Path to the folder to compress.
    :param output_zip: Path for the resulting zip file.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create the full path to the file and its relative path for the zip archive
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder_path)
                # Write the file to the zip archive
                zipf.write(full_path, relative_path)
    print(f"Folder compressed successfully into {output_zip}")


if __name__ == "__main__":
    # Example Usage
    folder_to_compress = "Results/March07Reports"  # Replace with your folder path
    output_zip_file = "1.zip"  # Replace with desired output file name or path
    compress_folder(folder_to_compress, output_zip_file)

    