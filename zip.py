import os
import zipfile


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


if __name__ == "__main__":
    # Example Usage
    folder_to_compress = "Results"  # Replace with your folder path
    output_zip_file = "results.zip"  # Replace with desired output file name or path
    compress_folder(folder_to_compress, output_zip_file)