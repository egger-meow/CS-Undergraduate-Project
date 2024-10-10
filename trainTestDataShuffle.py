import os
import shutil
import random

# Define source and destination directories
sourceDir = '"D:/leveling/leveling_data/v1/Normal/source"'
testDir = 'D:/leveling/leveling_data/v1/Normal/test'
trainDir = 'D:/leveling/leveling_data/v1/Normal/train'

testFileNum = 40

# Function to clear a directory
def clear_directory(directory):
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
files = os.listdir(source_dir)

# Randomly select 50 files to move to directory A
selected_files = random.sample(files, testFileNum)

# Move selected files to directory A
for file in selected_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(testDir, file))

# Move the remaining files to directory B
remaining_files = [file for file in files if file not in selected_files]
for file in remaining_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(trainDir, file))

print("Files have been cleared and moved successfully.")
