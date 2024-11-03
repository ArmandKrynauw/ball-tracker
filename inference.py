import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
from pathlib import Path
import torch

from utils import DATA_DIR, TEST_IMAGES_DIR, TRAIN_IMAGES_DIR


class PredictionVisualizer:
    def __init__(self, model_path, video_dir):
        self.model = YOLO(model_path)
        # self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.index = 0
        self.predictions = {}  # Cache for storing predictions

        # if len(self.image_paths) == 0:
        #     raise ValueError(f"No images found in {images_dir}")

        # print(f"Found {len(self.image_paths)} images. Use arrow keys to navigate, 'q' to quit.")

        results = self.model.track(
            video_dir,conf=0.1,max_det=1,save=True,project = Path(os.getcwd()) / "runs/detect" )
        # plt.ion()
        # self.fig = plt.figure(figsize=(12, 10))
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # self.plot_current_image()
        # plt.show(block=True)

    def get_prediction(self, image_path):
        if image_path not in self.predictions:
            results = self.model.predict(image_path, conf=0.1,max_det=1)
            self.predictions[image_path] = results[0]
        return self.predictions[image_path]

    def plot_current_image(self):
        plt.clf()

        image_path = self.image_paths[self.index]

        # Read and display image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        plt.imshow(image)

        # Get and plot predictions
        result = self.get_prediction(image_path)
        boxes = result.boxes

        for box in boxes:
            # Get box coordinates
            
            if torch.cuda.is_available():
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            else:
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
            h - 20,
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
    model_path = Path(os.getcwd()) / "runs/detect/run39/weights/best.pt"
    images_dir = TRAIN_IMAGES_DIR

    #Check if video exists
    if not os.path.exists("output_combined_video.mp4"):
        raise FileNotFoundError("output_combined_video.mp4 not found. Run inference)")

    PredictionVisualizer(model_path,"output_combined_video.mp4" )


if __name__ == "__main__":
    main()