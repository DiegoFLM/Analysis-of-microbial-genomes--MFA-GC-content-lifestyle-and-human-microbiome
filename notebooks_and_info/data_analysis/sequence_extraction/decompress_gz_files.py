import shutil
import gzip
import tarfile
from pathlib import Path


def decompress_gz_files(directory: Path, include_dirs: list):
    """
    Recursively decompresses .gz files found within a directory and its 
    specified subdirectories.

    Parameters:
    - directory: A pathlib.Path object representing the directory to search 
        for .gz files.
    - include_dirs: A list of directory names as strings where the function 
        should look for .gz files to decompress.
    """
    # Ensure the provided path is a directory
    if not directory.is_dir():
        print(f"The path {directory} is not a directory.")
        return

    # Iterate through each specified subdirectory
    for sub_dir_name in include_dirs:
        # Construct the subdirectory path
        sub_dir_path = directory / sub_dir_name
        
        # Check if the subdirectory exists
        if not sub_dir_path.exists() or not sub_dir_path.is_dir():
            print(f"The subdirectory {sub_dir_path} does not exist or is not a directory.")
            continue

        # Iterate recursively over all files with a .tgz extension in the subdirectory
        for gz_path in sub_dir_path.rglob('*.tgz'):
            # Define the output path by removing the .gz extension
            output_path = gz_path.with_suffix('')

            # Open the .gz file and the output file
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # print(f"Decompressed: at {sub_dir_name} to {output_path}")



def decompress_tgz_files(directory: Path, include_dirs: list):
    """
    Recursively decompresses .tgz files found within a directory and its 
    specified subdirectories.

    Parameters:
    - directory: A pathlib.Path object representing the directory to search 
        for .tgz files.
    - include_dirs: A list of directory names as strings where the function 
        should look for .tgz files to decompress.
    """
    # Ensure the provided path is a directory
    if not directory.is_dir():
        print(f"The path {directory} is not a directory.")
        return

    # Iterate through each specified subdirectory
    for sub_dir_name in include_dirs:
        # Construct the subdirectory path
        sub_dir_path = directory / sub_dir_name
        
        # Check if the subdirectory exists
        if not sub_dir_path.exists() or not sub_dir_path.is_dir():
            print(f"The subdirectory {sub_dir_path} does not exist or is not a directory.")
            continue

        # Iterate recursively over all files with a .tgz extension in the subdirectory
        for tgz_path in sub_dir_path.rglob('*.tgz'):
            try:
                # Open the .tgz file using tarfile
                with tarfile.open(tgz_path, 'r:gz') as tar:
                    # Extract all files to the same directory as the .tgz file
                    tar.extractall(path=tgz_path.parent)
                # print(f"Extracted: {tgz_path}")
            except Exception as e:
                print(f"Failed to extract {tgz_path}: {e}")
