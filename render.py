import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance, ImageDraw
from ultralytics import YOLO
import numpy as np

from utils import DATA_DIR, TEST_IMAGES_DIR, TRAIN_IMAGES_DIR

import concurrent.futures
from itertools import islice

def frames_to_video(frame_paths, model_path, output_path, max_frames, fps=25, contrast_factor=1.5, batch_size=10):
    """Converts a sequence of image frames to a video, applying contrast enhancement within bounding boxes in parallel batches."""

    # Initialize video writer based on the first frame's dimensions
    frame_sample = Image.open(frame_paths[0])
    width, height = frame_sample.size
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Function to process each frame
    def process_frame(frame_path):
        # Open the image as a PIL Image
        image = Image.open(frame_path)

        # Get the bounding boxes from YOLO predictions (simulating YOLO predictions here)
        result = YOLO(model_path).predict(frame_path, conf=0.1)[0].cpu()
        boxes = result.boxes

        # Apply highlight to each bounding box in the frame
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                image = highlight_bounding_box(image, (x1, y1, x2, y2), contrast_factor)

        # Convert the modified PIL image to a format compatible with VideoWriter
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame

    # Process frames in parallel batches
    print("Exporting video:")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        num_frames = min(len(frame_paths), max_frames)
        for i in range(0, num_frames, batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            print(f"Processing frames [{i} - {i + len(batch_paths) - 1}] of {num_frames}")

            # Submit batch processing tasks and gather results
            frames = list(executor.map(process_frame, batch_paths))

            # Write each processed frame to the video
            for frame in frames:
                video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")


def highlight_bounding_box(image, bounding_box, contrast_factor=1.5):
    """Increases the contrast of pixels within a bounding box to better highlight the object."""
    # image.width
    # image.height
    x1, y1, x2, y2 = map(int, bounding_box)
    print("COORDS: ", x1, y1, x2, y2)
    cropped = image.crop((x1, y1, x2, y2))

    # Create circular mask
    mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, x2 - x1, y2 - y1), fill=127)

    # Overlay mask on cropped image
    yellow = Image.new("RGB", cropped.size, "yellow")
    # enhanced_crop = ImageEnhance.Contrast(cropped).enhance(contrast_factor)
    cropped.paste(yellow, mask=mask)

    # Paste the enhanced crop back into the image
    image.paste(cropped, (x1, y1))
    return image


class PredictionVisualizer:
    def __init__(self, model_path, images_dir, visualize):
        self.model = YOLO(model_path)
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.index = 0
        self.predictions = {}  # Cache for storing predictions

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

        print(f"Found {len(self.image_paths)} images. Use arrow keys to navigate, 'q' to quit.")

        if visualize:
            plt.ion()
            self.fig = plt.figure(figsize=(12, 10))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.plot_current_image()
            plt.show(block=True)

    def get_prediction(self, image_path):
        if image_path not in self.predictions:
            results = self.model.predict(image_path, conf=0.1)
            self.predictions[image_path] = results[0]
        return self.predictions[image_path]

    def plot_current_image(self):
        plt.clf()

        image_path = self.image_paths[self.index]
        result = self.get_prediction(image_path).cpu()
        boxes = result.boxes

        # Read image and apply highlight on bounding boxes
        image = Image.open(image_path)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            image = highlight_bounding_box(image, (x1, y1, x2, y2))

        # Convert back to cv2 format for display
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        plt.imshow(image)


        if False:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]

                # Plot rectangle
                rect = Rectangle(
                    (x1, y1), 
                    x2 - x1, 
                    y2 - y1, 
                    fill=False, 
                    color="red", 
                    linewidth=2
                )
                plt.gca().add_patch(rect)

                # Add confidence score
                conf = box.conf[0]
                cls = int(box.cls[0])
                plt.text(
                    x1, 
                    y1 - 15,
                    f'Class {cls} ({conf:.2f})',
                    color="red",
                    fontweight="bold"
                )

        # Add image counter and filename
        plt.text(
            10, 
            30,
            f"Image {self.index + 1}/{len(self.image_paths)}",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6)
        )
        plt.text(
            10,
            image.width - 20,
            # image.shape[0] - 20,
            Path(image_path).name,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6)
        )

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


def main():
    model_path = DATA_DIR / "best.pt"
    images_dir = DATA_DIR / "dataset" / "train" / "images"
    video_output_path = DATA_DIR / "tracked_hockey_ball.mp4"

    VISUALIZE = False

    # Initialize the visualizer and render images with highlights
    visualizer = PredictionVisualizer(model_path, images_dir, VISUALIZE)

    if not VISUALIZE:
        # Export frames to video
        frames_to_video(visualizer.image_paths, model_path, video_output_path, 100)


if __name__ == "__main__":
    main()
