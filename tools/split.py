import os
import shutil
from sklearn.model_selection import train_test_split

# Paths for the input and output folders
annotations_folder = "annotations/dataset"
output_folder = "data/dataset"

# Directories for images and labels
images_folder = os.path.join(annotations_folder, "images")
labels_folder = os.path.join(annotations_folder, "labels")

# # Create output directories
# for folder in ["images", "labels"]:
#     for split in ["train", "test", "val"]:
#         os.makedirs(os.path.join(output_folder, folder, split), exist_ok=True)

# Get list of files
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))

# Split the data into train, test, and validation sets
train_images, test_images, train_labels, test_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.25, random_state=42
)

# Function to copy files to the new folders
def copy_files(file_list, source_folder, dest_folder):
    for file_name in file_list:
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(dest_folder, file_name))

# Copy images and labels to their respective folders
copy_files(train_images, images_folder, os.path.join(output_folder, "images", "train"))
copy_files(val_images, images_folder, os.path.join(output_folder, "images", "valid"))
copy_files(test_images, images_folder, os.path.join(output_folder, "images", "test"))
copy_files(train_labels, labels_folder, os.path.join(output_folder, "labels", "train"))
copy_files(val_labels, labels_folder, os.path.join(output_folder, "labels", "valid"))
copy_files(test_labels, labels_folder, os.path.join(output_folder, "labels", "test"))

print("Data split and copied successfully!")
