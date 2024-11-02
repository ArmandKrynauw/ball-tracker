from pathlib import Path
import subprocess
import cv2
from joblib import Parallel, delayed
import logging
import sys
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_clip_data(base_dir):
    """
    Load and process the clips.csv and map.csv files
    Returns a dictionary mapping video IDs to their frame ranges
    """
    clips_df = pd.read_csv(base_dir / 'annotations' / 'clips.csv')
    map_df = pd.read_csv(base_dir / 'annotations' / 'map.csv')

    # Create a mapping of video IDs to their frame ranges
    video_frames = {}
    for _, row in clips_df.iterrows():
        video_id = row['id']
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append((row['start_frame'], row['end_frame']))

    return video_frames

def extract_frames_cv2(video_path, output_dir, frame_ranges):
    """
    Extract specific frame ranges from video using OpenCV.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise Exception("Error opening video file")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        logging.info(f"Video {video_path.name}: {total_frames} frames, {fps} FPS")

        frame_count = 0

        # Process each frame range
        for start_frame, end_frame in frame_ranges:
            if start_frame > total_frames:
                logging.error(f"Start frame {start_frame} exceeds video length for {video_path}")
                continue

            logging.info(f"Extracting frames {start_frame} to {end_frame} from {video_path.name}")

            # Set position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

            # Read and save only frames within the range
            while frame_count <= (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame {start_frame + frame_count} from {video_path.name}")
                    break

                current_frame = start_frame + frame_count
                # Save frame
                output_path = output_dir / f"{video_path.stem}_{current_frame}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if frame_count % 100 == 0:  # Log progress every 100 frames
                    logging.info(f"Processed frame {current_frame} of {video_path.name}")

                frame_count += 1

            frame_count = 0  # Reset frame count for next range

        cap.release()
        return True

    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        return False

def process_video(video_path, output_dir, frame_ranges):
    """
    Process a single video file using OpenCV
    """
    logging.info(f"Processing {video_path} for frames {frame_ranges}")
    return extract_frames_cv2(video_path, output_dir, frame_ranges)

def main():
    # Set up paths
    base_dir = Path('data/field_hockey')
    videos_dir = base_dir / 'videos'
    frames_dir = base_dir / 'frames'

    # Load clip data
    video_frames = load_clip_data(base_dir)

    # Create output directory if it doesn't exist
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Prepare processing tasks
    tasks = []
    for video_id, frame_ranges in video_frames.items():
        print(f"Queued video {video_id} for processing")
        video_path = videos_dir / f"{video_id}.mp4"
        if video_path.exists():
            tasks.append((video_path, frames_dir, frame_ranges))
        else:
            logging.error(f"Video file not found: {video_path}")

    if not tasks:
        logging.error("No video files found to process")
        return

    # Process videos in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_video)(video_path, output_dir, frame_ranges) 
        for video_path, output_dir, frame_ranges in tasks
    )

    # Report results
    successful = sum(1 for r in results if r)
    total = len(tasks)
    logging.info(f"Successfully processed {successful}/{total} videos")

if __name__ == "__main__":
    main()
