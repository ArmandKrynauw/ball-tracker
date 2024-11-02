from pathlib import Path
import subprocess
import logging
import sys
import json
import pandas as pd
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

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
    """Get the exact frame count from the annotation file"""
    json_path = base_dir / 'annotations' / f'{video_id}_{start_frame}-{end_frame}.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                if entry['file_upload'].endswith(f'{video_id}_{start_frame}-{end_frame}.mp4'):
                    return entry['annotations'][0]['result'][0]['value']['framesCount']
        logging.error(f"Could not find frame count in {json_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading {json_path}: {str(e)}")
        return None

def extract_frames_ffmpeg(video_path, frames_dir, start_frame, end_frame, frame_count):
    """
    Extract frames using FFmpeg with exact frame count matching
    """
    try:
        # Create clip-specific directory
        clip_dir = frames_dir / f"{video_path.stem}_{start_frame}-{end_frame}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Extracting {frame_count} frames for clip {start_frame}-{end_frame} from {video_path.name}")
        
        # Adjust start and end frames by +2 to align with annotations
        frame_offset = 4
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'trim=start_frame={start_frame+frame_offset-1}:end_frame={end_frame+frame_offset-1},setpts=PTS-STARTPTS',
            '-r', '25',  # Force 25 fps extraction
            '-frames:v', str(frame_count),  # Extract exact number of frames
            '-start_number', '1',  # Start from 1 for each clip
            '-q:v', '2',
            f'{str(clip_dir)}/frame_%04d.jpg'  # Use 4-digit padding
        ]
        
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
        if frame_count is None:
            logging.error(f"Could not get frame count for {video_path.stem} {start_frame}-{end_frame}")
            continue
            
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
        if video_id == 'fh_01':
            continue
        print(f"Queued video {video_id} for processing")
        video_path = videos_dir / f"{video_id}.mp4"
        if video_path.exists():
            tasks.append((video_path, frames_dir, frame_ranges[4:5], base_dir))
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