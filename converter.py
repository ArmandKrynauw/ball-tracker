import json
import re
from pathlib import Path


def extract_frame_range(filename):
    """Extract frame range from filename like 'fh_01_4729-4943.json'"""
    match = re.search(r"(\w+)_(\d+)-(\d+)\.json", filename)
    if match:
        video_name = match.group(1)
        start_frame = int(match.group(2))
        end_frame = int(match.group(3))
        return video_name, start_frame, end_frame
    return None

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

def process_annotation_file(json_path, output_dir):
    """Process a single annotation file and create YOLO format annotations"""
    # Read JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract video name and frame range from filename
    video_info = extract_frame_range(json_path.name)
    if not video_info:
        print(f"Could not parse filename: {json_path.name}")
        return

    video_name, start_frame, end_frame = video_info

    # Find the correct annotation entry for this clip
    clip_data = None
    for entry in data:
        if entry["file_upload"].endswith(f"{video_name}_{start_frame}-{end_frame}.mp4"):
            clip_data = entry
            break

    if not clip_data:
        print(f"No matching annotation found for {json_path.name}")
        return

    # Get the sequence data and frames count
    sequence_data = clip_data["annotations"][0]["result"][0]["value"]
    frames_count = sequence_data["framesCount"]
    sequence = sequence_data["sequence"]

    # Create frame to annotation mapping
    frame_annotations = {}

    # Map frame numbers to their annotations
    for frame_data in sequence:
        if frame_data.get("enabled", True):  # Only use enabled frames
            clip_frame_num = frame_data[
                "frame"
            ]  # Frame number within the clip (1-based)

            x_center = frame_data["x"]
            y_center = frame_data["y"]
            width = frame_data["width"]
            height = frame_data["height"]

            # Convert to YOLO format
            x_yolo, y_yolo, w_yolo, h_yolo = normalize_to_yolo(
                x_center, y_center, width, height
            )
            frame_annotations[clip_frame_num] = (
                f"0 {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}"
            )

    # Create annotation files for all frames
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a clip-specific directory
    clip_dir = output_dir / f"{video_name}_{start_frame}-{end_frame}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Creating {frames_count} annotation files for clip {video_name}_{start_frame}-{end_frame}"
    )

    # Create files for all frames (1 to frames_count)
    for frame_num in range(1, frames_count + 1):
        # Create the annotation filename with new naming convention
        annotation_file = clip_dir / f"frame_{frame_num:04d}.txt"

        # Write the annotation (or empty file if no annotation exists)
        with open(annotation_file, "w") as f:
            if frame_num in frame_annotations:
                f.write(frame_annotations[frame_num])


def main():
    # Set up paths
    base_dir = Path("data/field_hockey")
    annotations_dir = base_dir / "annotations"
    output_dir = base_dir / "yolo_annotations"

    # Process each JSON file in the annotations directory
    for json_path in annotations_dir.glob("*.json"):
        print(f"Processing {json_path.name}")
        process_annotation_file(json_path, output_dir)


if __name__ == "__main__":
    main()
