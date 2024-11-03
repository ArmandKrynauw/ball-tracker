# %%==================== PREDICITON & DISPLAY FUNCTIONS ====================%%
import concurrent.futures
import glob
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from tqdm.notebook import tqdm
from ultralytics import YOLO
from matplotlib.patches import Rectangle
from utils import DATA_DIR, TEST_IMAGES_DIR, TRAIN_IMAGES_DIR
import concurrent.futures
from itertools import islice

import logging
import json
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Generates predictions for all frames and stores them in a dictionary.
def generate_predictions(model_path, image_paths, max_frames = None):
    model = YOLO(model_path)
    predictions = {}

    # Initialize progress bar
    cut_paths = image_paths
    if max_frames is not None:
        cut_paths = image_paths[:max_frames]

    for index, image_path in tqdm(enumerate(cut_paths), total=len(cut_paths), desc="Generating Predictions"):
        results = model.predict(image_path, conf=0.2, max_det=1)[0]
        boxes = results.cpu().boxes
        if boxes:
            # Store bounding boxes in the desired format
            predictions[index] = {
                "boxes": [tuple(map(int, boxes[0].xyxy[0].tolist()))]
            }

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

# Converts a sequence of image frames to a video, applying contrast enhancement within bounding boxes in parallel batches.
def frames_to_video(frame_paths, predictions, output_path, max_frames, fps=25, contrast_factor=1.5, batch_size=20):
    frame_sample = Image.open(frame_paths[0])
    width, height = frame_sample.size
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def process_frame(frame_index):
        image = Image.open(frame_paths[frame_index])
        boxes = predictions.get(frame_index)

        if boxes:
            for i, box in enumerate(boxes['boxes']):
                x1, y1, x2, y2 = box
                image = highlight_bounding_box(image, (x1, y1, x2, y2), i == 0, 1.5, contrast_factor)

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame

    print("Exporting video:")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        num_frames = len(frame_paths)
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

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

def highlight_bounding_box(image, bounding_box, draw_crosshairs=True, scale=1.25, contrast_factor=1.5):
    x1, y1, x2, y2 = map(int, bounding_box)

    # Get bounding box region
    cropped = image.crop((x1, y1, x2, y2))

    # Create mask
    mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0.1*(x2-x1), 0.1*(y2-y1), 0.9*(x2-x1), 0.9*(y2-y1)), fill=150)
    mask = mask.filter(ImageFilter.GaussianBlur(1))

    # mask = cropped.convert("L")

    # Draw overlay
    overlay = cropped
    color = Image.new("RGB", overlay.size, "yellow")
    overlay.paste(color, mask=mask)

    nx1 = int( max(x1 - (x2-x1)*(scale-1), 0) )
    ny1 = int( max(y1 - (y2-y1)*(scale-1), 0) )
    nx2 = int( min(x2 + (x2-x1)*(scale-1), image.width) )
    ny2 = int( min(y2 + (y2-y1)*(scale-1), image.height) )

    # print(nx1, ny1, nx2, ny2)

    overlay = overlay.resize( (nx2-nx1, ny2-ny1) )

    # Paste the enhanced crop back into the image
    # image.paste(cropped, int( x1 - (x2-x1)*(scale-1) ), int( y1 - (y2-y1)*(scale-1) ))
    image.paste(overlay, (nx1, ny1))


    if draw_crosshairs:
        padding = 10
        draw = ImageDraw.Draw(image)
        (bx1, by1, bx2, by2) = (
            max(0, min(image.width,     (nx1 - padding))),
            max(0, min(image.height,    (ny1 - padding))),
            max(0, min(image.width,     (nx2 + padding))),
            max(0, min(image.height,    (ny2 + padding)))
        )
        draw.line((bx1, by1, bx1 + (bx2 - bx1) / 2, by1), fill="red", width=2)
        draw.line((bx1, by1, bx1, by1 + (by2 - by1) / 2), fill="red", width=2)
        draw.line((bx2, by2, bx2, by2 - (by2 - by1) / 2), fill="red", width=2)
        draw.line((bx2, by2, bx2 - (bx2 - bx1) / 2, by2), fill="red", width=2)

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
        boxes = self.predictions.get(self.index)

        image = Image.open(image_path)
        if boxes:
            for box in boxes['boxes']:
                x1, y1, x2, y2 = box
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
    kf = initialize_kalman()
    filtered_predictions = {}
    previous_position = None

    for frame_index in sorted(predictions.keys()):
        if predictions.get(frame_index):
            box = predictions[frame_index]['boxes'][0]
            x1, y1, x2, y2 = box
            ball_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Center of bounding box

            if previous_position is not None and euclidean(ball_position, previous_position) > max_distance:
                ball_position = None
            else:
                kf.update(ball_position)
                previous_position = ball_position

        else:
            ball_position = None

        if ball_position is None:
            if previous_position is not None:
                kf.predict()
                predicted_position = kf.x[:2]
                ball_position = np.array(predicted_position).flatten()
            else:
                continue

        x, y = ball_position
        filtered_predictions[frame_index] = {
            'boxes': [(int(x - 10), int(y - 10), int(x + 10), int(y + 10))]
        }
    
    frame_indices = sorted(filtered_predictions.keys())
    for i in range(len(frame_indices) - 1):
        start_frame, end_frame = frame_indices[i], frame_indices[i + 1]
        gap = end_frame - start_frame

        if gap > 1 and gap <= max_gap:
            start_pos = np.array(filtered_predictions[start_frame]['boxes'][0][:2])
            end_pos = np.array(filtered_predictions[end_frame]['boxes'][0][:2])
            for j in range(1, gap):
                interpolated_pos = start_pos + (end_pos - start_pos) * (j / gap)
                interpolated_box = (int(interpolated_pos[0] - 10), int(interpolated_pos[1] - 10), int(interpolated_pos[0] + 10), int(interpolated_pos[1] + 10))
                filtered_predictions[start_frame + j] = {'boxes': [interpolated_box]}

    return filtered_predictions

# %%==================== KALMAN FILTER FUNCTIONS 2 ====================%%
import numpy as np
from scipy.spatial.distance import euclidean
from filterpy.kalman import KalmanFilter

# Filters and interpolates the predictions using a Kalman filter to smooth trajectories and fill in missing detections.
def filter_and_interpolate_predictions(predictions, max_gap=5, max_distance=50):
    kf = initialize_kalman()
    filtered_predictions = {}
    previous_position = None
    last_frame_with_detection = None

    for frame_index in sorted(predictions.keys()):
        if predictions[frame_index]:
            box = predictions[frame_index]['boxes'][0]
            x1, y1, x2, y2 = box
            ball_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Center of bounding box

            # Update the Kalman filter with the detected position
            kf.update(ball_position)
            filtered_predictions[frame_index] = {
                'boxes': [(int(x1), int(y1), int(x2), int(y2))]
            }
            previous_position = ball_position
            last_frame_with_detection = frame_index
        else:
            # If no detection, predict the next position
            if previous_position is not None:
                kf.predict()
                predicted_position = kf.x[:2]
                filtered_predictions[frame_index] = {
                    'boxes': [(int(predicted_position[0] - 10), int(predicted_position[1] - 10),
                                int(predicted_position[0] + 10), int(predicted_position[1] + 10))]
                }

    # Interpolate missing predictions if there are gaps
    frame_indices = sorted(filtered_predictions.keys())
    for i in range(len(frame_indices) - 1):
        start_frame, end_frame = frame_indices[i], frame_indices[i + 1]
        gap = end_frame - start_frame

        if gap > 1 and gap <= max_gap and last_frame_with_detection is not None:
            start_pos = np.array(filtered_predictions[last_frame_with_detection]['boxes'][0][:2])
            end_pos = np.array(filtered_predictions[end_frame]['boxes'][0][:2]) if end_frame in filtered_predictions else None

            if end_pos is not None:
                for j in range(1, gap):
                    interpolated_pos = start_pos + (end_pos - start_pos) * (j / (gap + 1))
                    interpolated_box = (int(interpolated_pos[0] - 10), int(interpolated_pos[1] - 10),
                                        int(interpolated_pos[0] + 10), int(interpolated_pos[1] + 10))
                    filtered_predictions[start_frame + j] = {'boxes': [interpolated_box]}

    return filtered_predictions



# %%==================== GENERATE PREDICTIONS ====================%%
model_path = DATA_DIR / "best_fh_2.pt"
# model_path = DATA_DIR / "best_iceHockey.pt"
# model_path = DATA_DIR / "best.pt"

# images_dir = DATA_DIR / "dataset" / "train" / "images"
# video_input_path = DATA_DIR / "field_hockey" / "videos" / "fh_04.mp4"
video_input_path = DATA_DIR / "testing" / "04" / "fh_04_737-945.mp4"
video_output_path = DATA_DIR / "testing" / "04" / "fh_04_737-945_tracked.mp4"
images_dir = DATA_DIR / "testing" / "04" / "images" / "737"

MAX_FRAMES = None
FPS = 25

# Extract frames from video
# video_to_frames(video_input_path, images_dir, start_frame=450, end_frame=720, fps=FPS)
video_to_frames(video_input_path, images_dir, start_frame=0, end_frame=MAX_FRAMES, fps=FPS)

# Get list of image paths
image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

# Generate predictions at the beginning
predictions = generate_predictions(model_path, image_paths, MAX_FRAMES)

# frames_to_video(image_paths, predictions, video_output_path, MAX_FRAMES)
# %%==================== APPLY KALMAN FILTER ====================%%
# kalman_predictions = filter_and_interpolate_predictions(predictions)

# %%==================== APPLY NN PREDICTOR ====================%%
# Load from .pth
# from pred import PositionPredictor, model_save_file
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = PositionPredictor()
# model.load_state_dict(torch.load(model_save_file, map_location=device))

# # Fill all missing frames
# nn_predictions = {}
# for frame_index in range(100):
#     if frame_index in predictions:
#         continue

#     accum = []
#     # Get previous 5 frames
#     for i in range(frame_index):
#         if frame_index - i in kalman_predictions
# %%==================== APPLY RANDOM TRACKER ====================%%
# from tracker import filter_and_interpolate_predictions as tracker_filter
# trakcer_predicitons = tracker_filter(converted_predictions, image_paths)

# %%==================== VISUALIZE ====================%%
# visualizer = PredictionVisualizer(image_paths, predictions)

# %%==================== CREATE VIDEO ====================%%
frames_to_video(image_paths, predictions, video_output_path, MAX_FRAMES)
