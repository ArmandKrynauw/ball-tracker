import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class YOLOVisualizer:
    def __init__(self, images_dir, labels_dir):
        # Get all image and label paths
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

        # Match images with their corresponding labels
        self.matched_pairs = []
        for img_path in image_paths:
            img_name = Path(img_path).stem
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            if os.path.exists(label_path):
                self.matched_pairs.append((img_path, label_path))

        if len(self.matched_pairs) == 0:
            raise ValueError("No matching image-label pairs found!")

        self.index = 0
        print(
            f"Found {len(self.matched_pairs)} matching image-label pairs. "
            f"Use arrow keys to navigate, 'q' to quit."
        )

        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.plot_current_image()
        plt.show(block=True)

    def plot_current_image(self):
        plt.clf()  # Clear the current figure

        image_path, label_path = self.matched_pairs[self.index]
        image_path = Path(image_path)
        label_path = Path(label_path)

        # Display current image name
        print(
            f"Showing pair {self.index + 1}/{len(self.matched_pairs)}: {image_path.name}"
        )

        # Read and display image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        plt.imshow(image)

        # Plot YOLO labels
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x, y, width, height = map(float, line.split())

                # Convert normalized YOLO coordinates to pixel coordinates
                x1 = int((x - width / 2) * w)
                y1 = int((y - height / 2) * h)
                x2 = int((x + width / 2) * w)
                y2 = int((y + height / 2) * h)

                # Plot rectangle
                rect = Rectangle(
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
            f"Image {self.index + 1}/{len(self.matched_pairs)}",
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
            self.index = (self.index + 1) % len(self.matched_pairs)
            self.plot_current_image()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.matched_pairs)
            self.plot_current_image()
        elif event.key == "q":
            plt.close("all")


def main():
    # Set up paths
    base_dir = Path('data/field_hockey')
    images_dir = base_dir / "frames"
    labels_dir = base_dir / "yolo_annotations"

    YOLOVisualizer(images_dir, labels_dir)


if __name__ == "__main__":
    main()
