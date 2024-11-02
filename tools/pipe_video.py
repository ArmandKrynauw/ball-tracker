import cv2
import numpy as np

def convert_video_and_detect_edges(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    # Get the video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter with grayscale settings
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Canny edge detection
        edges_frame = cv2.Canny(gray_frame, threshold1=-100, threshold2=200 )
        # Otsu's Thresholding
        _, otsu_thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # Adaptive Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_frame, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  
            2    
        )


        out.write(adaptive_thresh)



    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")

# Usage example
input_video = 'videos/Hockey1.mp4'  # Replace with your input video file path
output_video = 'output_combined_video.mp4'  # Replace with your desired output file path
convert_video_and_detect_edges(input_video, output_video)
