import os

def update_labels(directory):
    # Define the subdirectories to check
    subdirs = ["train", "test", "val"]

    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            print(f"Subdirectory '{subdir}' does not exist in {directory}. Skipping.")
            continue

        # Iterate over all files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the filename starts with 'akbars' and has a '.txt' extension
            if filename.startswith("akbars") and filename.endswith(".txt"):
                file_path = os.path.join(subdir_path, filename)

                # Read the contents of the label file
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Modify the class labels in the file contents
                modified_lines = []
                for line in lines:
                    parts = line.strip().split()
                    # Check if the class is 0 and change it to 1
                    if parts[0] == "0":
                        parts[0] = "1"
                    modified_lines.append(" ".join(parts))

                # Write the modified contents back to the file
                with open(file_path, 'w') as file:
                    file.write("\n".join(modified_lines) + "\n")

        print(f"Completed updates for '{subdir}' folder.")


# Provide the path to your directory here
directory = "data/dataset/labels"
update_labels(directory)

