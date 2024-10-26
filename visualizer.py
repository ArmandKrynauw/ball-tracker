import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class YOLOVisualizer:
    def __init__(self, images_dir, labels_dir):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
        self.index = 0

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

        if len(self.image_paths) != len(self.label_paths):
            raise ValueError("Number of images and labels do not match!")

        print(
            f"Found {len(self.image_paths)} images. Use arrow keys to navigate, 'q' to quit."
        )

        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.plot_current_image()
        plt.show(block=True)

    def plot_current_image(self):
        plt.clf()  # Clear the current figure

        image_path = Path(self.image_paths[self.index])
        label_path = Path(self.label_paths[self.index])

        # Display current image name
        print(
            f"Showing image {self.index + 1}/{len(self.image_paths)}: {image_path.name}"
        )

        # Read and display image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        plt.imshow(image)

        # Plot YOLO labels
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x, y, width, height = map(float, line.split())

                    # Convert normalized YOLO coordinates to pixel coordinates
                    x1 = int((x - width / 2) * w)
                    y1 = int((y - height / 2) * h)
                    x2 = int((x + width / 2) * w)
                    y2 = int((y + height / 2) * h)

                    # Plot rectangle
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=1
                    )
                    plt.gca().add_patch(rect)

                    # Add class label
                    plt.text(
                        x1,
                        y1 - 5,
                        f"Class {int(class_id)}",
                        color="red",
                        fontweight="bold",
                    )

        # Add image counter and filename
        plt.text(
            10,
            30,
            f"Image {self.index + 1}/{len(self.image_paths)}",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        plt.text(
            10,
            h - 20,
            image_path.name,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6),
        )

        plt.axis("off")
        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot

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
    images_dir = "data/dataset/images/valid"  # Change this to your images directory
    labels_dir = "data/dataset/labels/valid"  # Change this to your labels directory

    YOLOVisualizer(images_dir, labels_dir)


if __name__ == "__main__":
    main()
