import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def calculate_25fps_frame_count(start_frame, end_frame):
    """Calculate number of frames at 25fps given original 29.97fps frame range"""
    original_fps = 29.97
    target_fps = 25.0

    # Calculate duration in seconds
    duration = (end_frame - start_frame) / original_fps

    # Calculate number of frames at 25fps
    frame_count = int(duration * target_fps)

    return frame_count

def load_clip_data(base_dir):
    """Load and process the clips.csv"""
    clips_df = pd.read_csv(base_dir / 'annotations' / 'clips.csv')

    video_frames = {}
    for _, row in clips_df.iterrows():
        video_id = row['id']
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append((row['start_frame'], row['end_frame']))

    return video_frames

def get_frame_count(base_dir, video_id, start_frame, end_frame):
    """Get frame count from annotation file or calculate it"""
    json_path = base_dir / 'annotations' / f'{video_id}_{start_frame}-{end_frame}.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                if entry['file_upload'].endswith(f'{video_id}_{start_frame}-{end_frame}.mp4'):
                    return entry['annotations'][0]['result'][0]['value']['framesCount']
    except Exception:
        logging.info(f"No annotation file found for {video_id}_{start_frame}-{end_frame}, calculating frame count")

    # If we couldn't get frame count from annotation, calculate it
    return calculate_25fps_frame_count(start_frame, end_frame)

def extract_frames_ffmpeg(video_path, frames_dir, start_frame, end_frame, frame_count):
    """
    Extract frames using FFmpeg with exact frame count matching
    """
    try:
        # Create clip-specific directory
        clip_dir = frames_dir / f"{video_path.stem}_{start_frame}-{end_frame}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Extracting {frame_count} frames for clip {start_frame}-{end_frame} from {video_path.name}")

        print('clip_dir',clip_dir)
        frame_offset = 0
        # print('params',video_path,start_frame+frame_offset,end_frame+frame_offset,frame_count)
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'trim=start_frame={start_frame+frame_offset}:end_frame={end_frame+frame_offset},setpts=PTS-STARTPTS,fps=fps=25:round=down',
            '-frames:v', str(frame_count),
            '-start_number', '1',
            '-q:v', '2',
            f'{str(clip_dir)}/frame_%04d.jpg'
        ]
        print(' '.join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"FFmpeg failed: {result.stderr}")
            return False

        # Verify frame count
        extracted_frames = list(clip_dir.glob('frame_*.jpg'))
        if len(extracted_frames) != frame_count:
            logging.error(f"Frame count mismatch: expected {frame_count}, got {len(extracted_frames)}")
            return False

        return True

    except Exception as e:
        logging.error(f"Error processing {video_path} frames {start_frame}-{end_frame}: {str(e)}")
        return False

def process_video(video_path, frames_dir, frame_ranges, base_dir):
    """
    Process a single video file
    """
    results = []
    for start_frame, end_frame in frame_ranges:
        logging.info(f"Processing {video_path.name} frames {start_frame}-{end_frame}")

        frame_count = get_frame_count(base_dir, video_path.stem, start_frame, end_frame)
        result = extract_frames_ffmpeg(video_path, frames_dir, start_frame, end_frame, frame_count)
        results.append(result)

    return all(results)  # Return True only if all clips were successful

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
            tasks.append((video_path, frames_dir, frame_ranges, base_dir))
        else:
            logging.error(f"Video file not found: {video_path}")

    if not tasks:
        logging.error("No video files found to process")
        return

    # Process videos in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_video)(video_path, output_dir, frame_ranges, base_dir)
        for video_path, output_dir, frame_ranges, base_dir in tasks
    )

    # Report results
    successful = sum(1 for r in results if r)
    total = len(tasks)
    logging.info(f"Successfully processed {successful}/{total} videos")

if __name__ == "__main__":
    main()
