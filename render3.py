# %%==================== PREDICITON & DISPLAY FUNCTIONS ====================%%
import glob
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from ultralytics import YOLO
import numpy as np
from utils import DATA_DIR, TEST_IMAGES_DIR, TRAIN_IMAGES_DIR
import concurrent.futures
from itertools import islice
from tqdm.notebook import tqdm

# Generates predictions for all frames and stores them in a dictionary.
def generate_predictions(model_path, image_paths, max_frames):
    model = YOLO(model_path)
    predictions = {}

    # Initialize progress bar
    for index, image_path in tqdm(enumerate(image_paths[:max_frames]), total=min(len(image_paths), max_frames), desc="Generating Predictions"):
        results = model.predict(image_path, conf=0.2, max_det=1)[0]
        boxes = results.cpu().boxes
        if boxes:
            # Store bounding boxes in the desired format
            predictions[index] = {
                "boxes": [tuple(map(int, boxes[0].xyxy[0].tolist()))]
            }

    return predictions

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
        num_frames = min(len(frame_paths), max_frames)
        for i in range(0, num_frames, batch_size):
            batch_indices = range(i, min(i + batch_size, num_frames))
            print(f"Processing frames [{i} - {i + len(batch_indices) - 1}] of {num_frames}")

            frames = list(executor.map(process_frame, batch_indices))

            for frame in frames:
                video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")

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
    color = Image.new("RGB", overlay.size, "magenta")
    overlay.paste(color, mask=mask)

    nx1 = int( max(x1 - (x2-x1)*(scale-1), 0) )
    ny1 = int( max(y1 - (y2-y1)*(scale-1), 0) )
    nx2 = int( min(x2 + (x2-x1)*(scale-1), image.width) )
    ny2 = int( min(y2 + (y2-y1)*(scale-1), image.height) )

    print(nx1, ny1, nx2, ny2)

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
model_path = DATA_DIR / "best.pt"
images_dir = DATA_DIR / "dataset" / "train" / "images"
video_output_path = DATA_DIR / "tracked_hockey_ball.mp4"

# Get list of image paths
image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

# Generate predictions at the beginning
predictions = generate_predictions(model_path, image_paths, 100)

# %%==================== APPLY KALMAN FILTER ====================%%
kalman_predictions = filter_and_interpolate_predictions(predictions)

# %%==================== APPLY NN PREDICTOR ====================%%
# Load from .pth
from pred import PositionPredictor, model_save_file
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PositionPredictor()
model.load_state_dict(torch.load(model_save_file, map_location=device))

# Fill all missing frames
nn_predictions = {}
for frame_index in range(100):
    if frame_index in predictions:
        continue

    accum = []
    # Get previous 5 frames
    for i in range(frame_index):
        if frame_index - i in kalman_predictions

# %%==================== VISUALIZE ====================%%
visualizer = PredictionVisualizer(image_paths, predictions)

# %%==================== CREATE VIDEO ====================%%
frames_to_video(image_paths, predictions, video_output_path, 100)
