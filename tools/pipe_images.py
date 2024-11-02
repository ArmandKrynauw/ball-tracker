import cv2
import os

# Base input and output folders
input_folder = 'data/dataset/images'
output_folder = 'data/dataset/images'

# Create the base output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Walk through all subdirectories and files in the input folder
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not filename.startswith("gray_") and not filename.startswith("edges_"):
            # Construct full file paths for input and output
            img_path = os.path.join(root, filename)
            
            # Recreate the folder structure in the output folder
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            print(f"relative_path: {relative_path}")
            print(f"Output subfolder: {output_subfolder}")
            
            # Read the image
            image = cv2.imread(img_path)
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #     adaptive_thresh = cv2.adaptiveThreshold(
        #         gray_image, 
        #         255, 
        #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #         cv2.THRESH_BINARY, 
        #         11,  
        #         2    
        # )

            # edges = cv2.Canny(gray_image, threshold1=5, threshold2=150)
            
            # Save the edge-detected image in the corresponding subfolder of the output directory
            edges_output_path = os.path.join(output_subfolder, f"{filename}")
            cv2.imwrite(edges_output_path, gray_image)
            print(f"Processed and saved edge-detected image: {edges_output_path}")
        else:
            print(f"Skipped non-image file: {filename}")
