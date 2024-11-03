import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class YOLOVisualizer:
    def __init__(self, frames_dir, labels_dir):
        # Get all clip directories
        frame_clip_dirs = sorted([d for d in frames_dir.glob("*") if d.is_dir()])
        label_clip_dirs = sorted([d for d in labels_dir.glob("*") if d.is_dir()])

        if len(frame_clip_dirs) == 0:
            raise ValueError(f"No frame clip directories found in {frames_dir}")

        # Match clips with their corresponding labels
        self.clips = []
        for frame_dir in frame_clip_dirs:
            clip_name = frame_dir.name
            label_dir = labels_dir / clip_name

            if label_dir.exists():
                # Get all frames and labels in this clip
                frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
                label_paths = sorted(label_dir.glob("frame_*.txt"))

                # Match frames with labels
                matched_pairs = []
                for frame_path in frame_paths:
                    frame_num = frame_path.stem
                    label_path = label_dir / f"{frame_num}.txt"
                    if label_path.exists():
                        matched_pairs.append((frame_path, label_path))

                if matched_pairs:
                    self.clips.append({
                        'name': clip_name,
                        'pairs': matched_pairs,
                        'current_index': 0
                    })

        if len(self.clips) == 0:
            raise ValueError(f"No matching clips found!\n\tFrames dir: {frames_dir}\n\tLabels dir: {labels_dir}")

        self.current_clip = 0
        print(
            f"Found {len(self.clips)} clips. "
            f"Use left/right arrows to navigate frames, up/down arrows to navigate clips, 'q' to quit."
        )

        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.tight_layout(pad=0)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.plot_current_frame()
        plt.show(block=True)

    def plot_current_frame(self):
        plt.clf()  # Clear the current figure

        clip = self.clips[self.current_clip]
        frame_path, label_path = clip['pairs'][clip['current_index']]

        # Display current image name
        print(
            f"Showing clip {self.current_clip + 1}/{len(self.clips)}: {clip['name']}, "
            f"frame {clip['current_index'] + 1}/{len(clip['pairs'])}"
        )

        # Read and display image
        image = cv2.imread(str(frame_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Create axis that fills the whole figure
        ax = plt.gca()
        ax.set_position([0, 0, 1, 1])
        plt.imshow(image)

        # Plot YOLO labels if file is not empty
        if label_path.stat().st_size > 0:
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
                    ax.add_patch(rect)

                    # Add class label
                    plt.text(
                        x1,
                        y1 - 5,
                        f"Class {int(class_id)}",
                        color="red",
                        fontweight="bold",
                    )

        # Add clip and frame information
        plt.text(
            10,
            30,
            f"Clip {self.current_clip + 1}/{len(self.clips)} - Frame {clip['current_index'] + 1}/{len(clip['pairs'])}",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        plt.text(
            10,
            h - 20,
            f"{clip['name']} - {frame_path.name}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6),
        )

        plt.axis('off')  # Turn off axes
        self.fig.tight_layout(pad=0)  # Remove padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
        plt.draw()
        plt.pause(0.001)

    def on_key(self, event):
        clip = self.clips[self.current_clip]

        if event.key == "right":
            # Next frame in current clip
            clip['current_index'] = (clip['current_index'] + 1) % len(clip['pairs'])
        elif event.key == "left":
            # Previous frame in current clip
            clip['current_index'] = (clip['current_index'] - 1) % len(clip['pairs'])
        elif event.key == "up":
            # Previous clip
            self.current_clip = (self.current_clip - 1) % len(self.clips)
        elif event.key == "down":
            # Next clip
            self.current_clip = (self.current_clip + 1) % len(self.clips)
        elif event.key == "pageup":
            # Jump 10 clips backward
            self.current_clip = (self.current_clip - 10) % len(self.clips)
        elif event.key == "pagedown":
            # Jump 10 clips forward
            self.current_clip = (self.current_clip + 10) % len(self.clips)
        elif event.key == "home":
            # Jump to first clip
            self.current_clip = 0
        elif event.key == "end":
            # Jump to last clip
            self.current_clip = len(self.clips) - 1
        elif event.key == "q":
            plt.close("all")
            return

        self.plot_current_frame()


def main():
    # Set up paths
    base_dir = Path('data/field_hockey')
    frames_dir = base_dir / "frames"
    labels_dir = base_dir / "yolo_annotations"

    YOLOVisualizer(frames_dir, labels_dir)


if __name__ == "__main__":
    main()
