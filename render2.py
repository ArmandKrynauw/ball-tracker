# %%==================== PREDICITON & DISPLAY FUNCTIONS ====================%%
import concurrent.futures
import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm.notebook import tqdm
from ultralytics import YOLO

from utils import DATA_DIR

import logging
import json
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def generate_predictions(model_path, image_paths, max_frames):
    """Generates predictions for all frames and stores them in a dictionary."""
    model = YOLO(model_path)
    predictions = {}

    # Initialize progress bar
    for index, image_path in tqdm(enumerate(image_paths[:max_frames]), total=min(len(image_paths), max_frames), desc="Generating Predictions"):
        results = model.predict(image_path, conf=0.2, max_det=1)
        predictions[index] = results[0]  # Store prediction by frame number

    return predictions

def convert_yolo_predictions(predictions):
    """
    Convert YOLO predictions to the format used by tracking filters.

    Args:
        predictions (dict): Dictionary of YOLO predictions indexed by frame number

    Returns:
        dict: Predictions in tracking filter format with structure:
            {frame_index: {'boxes': [{'xyxy': np.array([x1, y1, x2, y2])}]}}
    """
    converted_predictions = {}

    for frame_index, pred in predictions.items():
        if pred.boxes and len(pred.boxes) > 0:
            # Get the first box (assuming single ball detection)
            box = pred.boxes[0].cpu()
            x1, y1, x2, y2 = box.xyxy[0]

            # Store in new format
            converted_predictions[frame_index] = {
                'boxes': [{
                    'xyxy': np.array([x1, y1, x2, y2])
                }]
            }
        else:
            # No detection for this frame
            converted_predictions[frame_index] = {
                'boxes': []
            }

    return converted_predictions

def frames_to_video(frame_paths, predictions, output_path, max_frames, fps=25, contrast_factor=1.5, batch_size=20):
    """Converts a sequence of image frames to a video, applying contrast enhancement within bounding boxes in parallel batches."""
    frame_sample = Image.open(frame_paths[0])
    width, height = frame_sample.size
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def process_frame(frame_index):
        image = Image.open(frame_paths[frame_index])
        boxes = predictions[frame_index]["boxes"]

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box["xyxy"]
                image = highlight_bounding_box(image, (x1, y1, x2, y2), i == 0, contrast_factor)

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame

    print("Exporting video:")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        num_frames = min(len(frame_paths), max_frames)
        for i in range(0, num_frames, batch_size):
            batch_indices = range(i, min(i + batch_size, num_frames))
            print(f"Processing frames [{i} - {i + len(batch_indices) - 1}] of {num_frames}")

            frames = list(executor.map(process_frame, batch_indices))

            for frame in frames:
                video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")

def video_to_frames(video_path, output_dir, fps=None, start_frame=None, end_frame=None):
    """
    Extract frames from a video using FFmpeg.

    Args:
        video_path (str or Path): Path to the input video file
        output_dir (str or Path): Directory to save extracted frames
        fps (float, optional): Frames per second to extract. If None, uses video's original FPS
        start_frame (int, optional): First frame to extract. If None, starts from beginning
        end_frame (int, optional): Last frame to extract. If None, processes until end

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video metadata using FFprobe
        probe_cmd = f'ffprobe -v quiet -print_format json -show_streams "{str(video_path)}"'
        print(f"Running: {probe_cmd}")

        metadata = os.popen(probe_cmd).read()
        metadata = json.loads(metadata)
        video_stream = next(s for s in metadata['streams'] if s['codec_type'] == 'video')

        # Get original FPS if not specified
        if fps is None:
            if 'avg_frame_rate' in video_stream:
                fps_num, fps_den = map(int, video_stream['avg_frame_rate'].split('/'))
                fps = fps_num / fps_den if fps_den != 0 else 25
            else:
                fps = 25

        # Build the FFmpeg filter string
        filter_parts = []

        # Add trim filter if start_frame or end_frame is specified
        if start_frame is not None or end_frame is not None:
            trim_parts = ['trim=']
            if start_frame is not None:
                trim_parts.append(f'start_frame={start_frame}')
            if end_frame is not None:
                if start_frame is not None:
                    trim_parts.append(':')
                trim_parts.append(f'end_frame={end_frame}')
            filter_parts.extend([
                ''.join(trim_parts),
                'setpts=PTS-STARTPTS'
            ])

        # Add fps filter
        filter_parts.append(f'fps={fps}')

        # Combine all filters
        filter_string = ','.join(filter_parts)

        # Calculate expected frame count if both start and end are specified
        frame_count = ''
        if start_frame is not None and end_frame is not None:
            frame_count = f'-frames:v {end_frame - start_frame + 1}'

        # Construct and run FFmpeg command
        ffmpeg_cmd = (
            f'ffmpeg -i "{str(video_path)}" '
            f'-vf "{filter_string}" '
            f'{frame_count} '
            f'-frame_pts 1 -q:v 2 "{str(output_dir)}/frame_%04d.jpg"'
        )

        print(f"Running: {ffmpeg_cmd}")
        os.system(ffmpeg_cmd)

        # Verify frames were extracted
        extracted_frames = list(output_dir.glob('frame_*.jpg'))
        frame_count = len(extracted_frames)

        if frame_count == 0:
            logging.error("No frames were extracted")
            return False

        logging.info(f"Successfully extracted {frame_count} frames to {output_dir}")
        return True

    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        return False

def highlight_bounding_box(image, bounding_box, draw_crosshairs=True, contrast_factor=1.5):
    x1, y1, x2, y2 = map(int, bounding_box)
    cropped = image.crop((x1, y1, x2, y2))
    mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, x2 - x1, y2 - y1), fill=200)

    yellow = Image.new("RGB", cropped.size, "magenta")
    cropped.paste(yellow, mask=mask)
    image.paste(cropped, (x1, y1))

    if draw_crosshairs:
        padding = 10
        draw = ImageDraw.Draw(image)
        (bx1, by1, bx2, by2) = (
            max(0, min(image.width, x1 - padding)),
            max(0, min(image.height, y1 - padding)),
            max(0, min(image.width, x2 + padding)),
            max(0, min(image.height, y2 + padding))
        )
        draw.line((bx1, by1, bx1 + (bx2 - bx1) / 2, by1), fill="yellow", width=2)
        draw.line((bx1, by1, bx1, by1 + (by2 - by1) / 2), fill="yellow", width=2)
        draw.line((bx2, by2, bx2, by2 - (by2 - by1) / 2), fill="yellow", width=2)
        draw.line((bx2, by2, bx2 - (bx2 - bx1) / 2, by2), fill="yellow", width=2)

    return image

class PredictionVisualizer:
    def __init__(self, image_paths, predictions):
        self.image_paths = image_paths
        self.predictions = predictions
        self.index = 0

        if len(self.image_paths) == 0:
            raise ValueError("No images found in the specified directory.")

        print(f"Found {len(self.image_paths)} images. Use arrow keys to navigate, 'q' to quit.")

        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.plot_current_image()
        plt.show(block=True)

    def plot_current_image(self):
        plt.clf()

        image_path = self.image_paths[self.index]
        result = self.predictions[self.index].cpu()
        boxes = result.boxes

        image = Image.open(image_path)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            image = highlight_bounding_box(image, (x1, y1, x2, y2))

        plt.imshow(image)
        plt.text(10, 30, f"Image {self.index + 1}/{len(self.image_paths)}", color="white", fontsize=12, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6))
        plt.text(10, image.width - 20, Path(image_path).name, color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.6))
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)

    def on_key(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.image_paths)
            self.plot_current_image()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.image_paths)
            self.plot_current_image()
        elif event.key == "q":
            plt.close("all")

# %%==================== KALMAN FILTER FUNCTIONS ====================%%
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import euclidean


def initialize_kalman():
    """Initialize a 2D Kalman filter for tracking the ball's position."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                     [0, 1, 0, 0]])
    kf.P *= 1000.  # Covariance matrix
    kf.R *= 10     # Measurement noise
    kf.Q = np.eye(4) * 0.01  # Process noise
    return kf

def filter_and_interpolate_predictions(predictions, max_gap=5, max_distance=50):
    """
    Filters and interpolates the predictions using a Kalman filter to
    smooth trajectories and fill in missing detections.

    Args:
        predictions (dict): Original predictions with frame indices as keys.
        max_gap (int): Maximum gap of frames to interpolate between.
        max_distance (float): Maximum distance to consider a detection valid.

    Returns:
        dict: New set of predictions, same format as input.
    """
    kf = initialize_kalman()
    filtered_predictions = {}
    previous_position = None

    for frame_index in sorted(predictions.keys()):
        # Check if a ball was detected in this frame
        if predictions[frame_index].boxes:
            box = predictions[frame_index].boxes[0].cpu()
            x1, y1, x2, y2 = box.xyxy[0]
            ball_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Center of bounding box

            # Discard detections too far from the predicted position
            if previous_position is not None and euclidean(ball_position, previous_position) > max_distance:
                ball_position = None
            else:
                # Update Kalman filter with the detected position
                kf.update(ball_position)
                previous_position = ball_position

        else:
            ball_position = None

        # If no ball detected, use prediction from Kalman filter
        if ball_position is None:
            if previous_position is not None:
                # Predict next position using Kalman filter
                kf.predict()
                predicted_position = kf.x[:2]
                ball_position = np.array(predicted_position).flatten()
            else:
                # Skip if there's no prior position for initialization
                filtered_predictions[frame_index] = None
                continue

        # Save the filtered/interpolated prediction for the frame
        filtered_predictions[frame_index] = {
            'boxes': [{'xyxy': np.array([ball_position[0]-10, ball_position[1]-10, ball_position[0]+10, ball_position[1]+10])}]
        }

    # Fill in gaps in detection using linear interpolation
    frame_indices = sorted(filtered_predictions.keys())
    for i in range(len(frame_indices) - 1):
        start_frame, end_frame = frame_indices[i], frame_indices[i + 1]
        gap = end_frame - start_frame

        if gap > 1 and gap <= max_gap:
            # Perform linear interpolation
            start_pos = np.array(filtered_predictions[start_frame]['boxes'][0]['xyxy'][:2])
            end_pos = np.array(filtered_predictions[end_frame]['boxes'][0]['xyxy'][:2])
            for j in range(1, gap):
                interpolated_pos = start_pos + (end_pos - start_pos) * (j / gap)
                interpolated_box = {
                    'xyxy': np.array([interpolated_pos[0] - 10, interpolated_pos[1] - 10, interpolated_pos[0] + 10, interpolated_pos[1] + 10])
                }
                filtered_predictions[start_frame + j] = {'boxes': [interpolated_box]}

    return filtered_predictions

# %%==================== GENERATE PREDICTIONS ====================%%
model_path = DATA_DIR / "best_fh_2.pt"
images_dir = DATA_DIR / "dataset" / "train" / "images"
video_output_path = DATA_DIR / "tracked_hockey_ball.mp4"
video_input_path = DATA_DIR / "field_hockey" / "videos" / "fh_04.mp4"

# Extract frames from video
video_to_frames(video_input_path, images_dir, start_frame=450, end_frame=720, fps=30)

# Get list of image paths
image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

# Generate predictions at the beginning
predictions = generate_predictions(model_path, image_paths, 100)

# %%==================== APPLY KALMAN FILTER ====================%%
kalman_predictions = filter_and_interpolate_predictions(predictions)

# %%==================== APPLY PARTICLE FILTER ====================%%
from particle_filter import filter_and_interpolate_predictions
particle_predictions = filter_and_interpolate_predictions(predictions)

# %%==================== VISUALIZE ====================%%
visualizer = PredictionVisualizer(image_paths, predictions)

# %%==================== CREATE VIDEO ====================%%
frames_to_video(image_paths, convert_yolo_predictions(predictions), video_output_path, 100, fps=30)