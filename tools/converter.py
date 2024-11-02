import os
import cv2

def get_frame_rate(video_path):
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get the frame rate
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    video_capture.release()  # Release the video capture object
    return fps

def create_directory(video_name, output_path):
    video_folder = os.path.join(output_path, video_name)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    return video_folder

def convert_video_to_frames(video_path, output_path, frame_interval=10):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = create_directory(video_name, output_path)

    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    frame_count = 0
    saved_frame_count = 0

    while success:
        # Save every N-th frame
        if frame_count % frame_interval == 0:
            frame_file = os.path.join(video_folder, f"frame_{saved_frame_count:04d}.png")
            cv2.imwrite(frame_file, frame)
            saved_frame_count += 1
        
        success, frame = video_capture.read()
        frame_count += 1

    video_capture.release()

def process_videos(input_folder, output_folder, frame_interval=10):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):
            video_path = os.path.join(input_folder, file_name)
            convert_video_to_frames(video_path, output_folder, frame_interval)



# frame_rate = get_frame_rate("videos/Hockey1.mp4")

# if frame_rate is not None:
#     print(f"The frame rate of the video is: {frame_rate} FPS")


# # Usage example: 
input_folder = 'videos'  # Folder containing .mov files
output_folder = 'videoImages'  # Folder where you want to save the images


# Note that both images are 25 fps
process_videos(input_folder, output_folder, frame_interval=10)

