import os
import shutil
import random

from settings import norm_sourceDir, abnorm_sourceDir, norm_trainDataDir, abnorm_trainDataDir, norm_testDataDir, abnorm_testDataDir

def list_all_files(src_dir):
    file_paths = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def cut(sourceDir, trainDir, testDir, testFileNum = 100):

    # Function to clear a directory
    def clear_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return 
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory

    # Clear destination directories
    clear_directory(testDir)
    clear_directory(trainDir)

    # Get all file names from the source directory
    files = list_all_files(sourceDir)
    # Randomly select testFileNum files to move to directory A
    selected_files = random.sample(files, testFileNum)

    # Move selected files to directory A
    for file in selected_files:
        file_name = os.path.basename(file)
        # Construct the destination path (flattened structure)
        dest_file_path = os.path.join(testDir, file_name)
        os.link(file, dest_file_path)
        # shutil.copy(os.path.join(sourceDir, file), os.path.join(testDir, file))

    # Move the remaining files to directory B
    remaining_files = [file for file in files if file not in selected_files]
    for file in remaining_files:
        file_name = os.path.basename(file)
        # Construct the destination path (flattened structure)
        dest_file_path = os.path.join(trainDir, file_name)
        os.link(file, dest_file_path)

    print("Files have been cleared and copied successfully.")

def main():
    cut(norm_sourceDir, norm_trainDataDir, norm_testDataDir)
    cut(abnorm_sourceDir, abnorm_trainDataDir, abnorm_testDataDir)

if __name__ == "__main__":  
    main()