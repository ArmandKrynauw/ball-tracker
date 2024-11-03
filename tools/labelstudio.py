import os
import shutil

# Set the path to your frames directory and the destination images folder
frames_dir = "data/field_hockey/yolo_annotations"
labels_dir = os.path.join(frames_dir, "labels")
destination_path = "data/field_hock"

# # Create the yolo_annotations directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Loop through each folder inside the frames directory
for folder_name in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, folder_name)
    
    print(f"Processing folder: {folder_path}")
    print(f"folder_name: {folder_name}")
    # Ensure we're working with directories only
    if os.path.isdir(folder_path) and folder_name != "yolo_annotations":
        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image with the "frame" naming format
            if filename.startswith("frame_"):
                # Construct the new file name with the folder name as a prefix
                new_name = f"{folder_name}_{filename}"
                
                # Get the full path to the old file and the destination file
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(labels_dir, new_name)
                
                # Rename and move the file to the yolo_annotations directory
                shutil.move(old_file, new_file)
                print(f"Moved and renamed {old_file} to {new_file}")


# final_destination = os.path.join(destination_path, "labels")
# print('final_destination:',final_destination)
# shutil.move(labels_dir, final_destination)