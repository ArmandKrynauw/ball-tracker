import json
from pathlib import Path

import pandas as pd


def normalize_to_yolo(
    x_tl_percent,  # x top-left percentage
    y_tl_percent,  # y top-left percentage
    width_percent,
    height_percent,
    img_width=1920,
    img_height=1080,
):
    """Convert from top-left percentage coordinates to YOLO format (center-based)"""
    # Convert width and height to pixels
    width_px = (width_percent / 100.0) * img_width
    height_px = (height_percent / 100.0) * img_height

    # Convert top-left percentages to pixels
    x_tl_px = (x_tl_percent / 100.0) * img_width
    y_tl_px = (y_tl_percent / 100.0) * img_height

    # Calculate center coordinates in pixels
    x_center_px = x_tl_px + (width_px / 2)  # Add half width to get to center
    y_center_px = y_tl_px + (height_px / 2)  # Add half height to get to center

    # Convert to YOLO format (normalized 0-1)
    x_yolo = x_center_px / img_width
    y_yolo = y_center_px / img_height
    w_yolo = width_px / img_width
    h_yolo = height_px / img_height

    return x_yolo, y_yolo, w_yolo, h_yolo


def calculate_25fps_frame_count(start_frame, end_frame):
    """Calculate number of frames at 25fps given original 29.97fps frame range"""
    original_fps = 29.97
    target_fps = 25.0

    # Calculate duration in seconds
    duration = (end_frame - start_frame) / original_fps

    # Calculate number of frames at 25fps
    frame_count = int(duration * target_fps)

    return frame_count


def process_clip(video_id, start_frame, end_frame, annotations_dir, output_dir):
    """Process a single clip, with or without annotations"""
    # Create clip directory
    clip_dir = output_dir / f"{video_id}_{start_frame}-{end_frame}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Calculate number of frames at 25fps
    frame_count = calculate_25fps_frame_count(start_frame, end_frame)

    # Check for annotation file
    json_path = annotations_dir / f"{video_id}_{start_frame}-{end_frame}.json"

    if json_path.exists():
        print(f"Processing annotations for clip {video_id}_{start_frame}-{end_frame}")
        # Process annotations
        with open(json_path, "r") as f:
            data = json.load(f)

        # Find the correct annotation entry
        clip_data = None
        for entry in data:
            if entry["file_upload"].endswith(f"{video_id}_{start_frame}-{end_frame}.mp4"):
                clip_data = entry
                break

        if clip_data:
            # Get sequence data
            sequence_data = clip_data["annotations"][0]["result"][0]["value"]
            sequence = sequence_data["sequence"]

            # Create frame to annotation mapping
            frame_annotations = {}
            for frame_data in sequence:
                if frame_data.get("enabled", True):
                    clip_frame_num = frame_data["frame"]
                    x_center = frame_data["x"]
                    y_center = frame_data["y"]
                    width = frame_data["width"]
                    height = frame_data["height"]

                    x_yolo, y_yolo, w_yolo, h_yolo = normalize_to_yolo(
                        x_center, y_center, width, height
                    )
                    frame_annotations[clip_frame_num] = (
                        f"0 {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}"
                    )
    else:
        print(f"No annotations found for clip {video_id}_{start_frame}-{end_frame}")
        frame_annotations = {}

    print(f"Creating {frame_count} annotation files for clip {video_id}_{start_frame}-{end_frame}")

    # Create files for all frames
    for frame_num in range(1, frame_count + 1):
        annotation_file = clip_dir / f"frame_{frame_num:04d}.txt"
        with open(annotation_file, "w") as f:
            if frame_num in frame_annotations:
                f.write(frame_annotations[frame_num])


def main():
    # Set up paths
    base_dir = Path("data/field_hockey")
    annotations_dir = base_dir / "annotations"
    output_dir = base_dir / "yolo_annotations"

    # Read clips.csv
    clips_df = pd.read_csv(base_dir / "annotations" / "clips.csv")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip from clips.csv
    for _, row in clips_df.iterrows():
        process_clip(
            row['id'],
            row['start_frame'],
            row['end_frame'],
            annotations_dir,
            output_dir
        )


if __name__ == "__main__":
    main()
