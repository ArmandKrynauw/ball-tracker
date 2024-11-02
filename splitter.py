from pathlib import Path
import subprocess
import logging
import sys
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

def extract_frames_ffmpeg(video_path, output_dir, frame_ranges):
    """
    Extract frames using FFmpeg with the same logic used for clip creation
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame interval in seconds (from annotations)
        frame_interval = 0.04
        
        for start_frame, end_frame in frame_ranges:
            start_time = (start_frame - 1) * frame_interval
            end_time = end_frame * frame_interval
            
            logging.info(f"Extracting frames {start_frame} to {end_frame} from {video_path.name}")
            logging.info(f"Time range: {start_time:.2f}s to {end_time:.2f}s")
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS',
                # '-vsync', '0',
                '-r', '25',  # Force 25 fps extraction (1/0.04)
                '-start_number', str(start_frame),
                '-q:v', '2',
                f'{str(output_dir)}/{video_path.stem}_%d.jpg'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"FFmpeg failed: {result.stderr}")
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        return False

def process_video(video_path, output_dir, frame_ranges):
    """
    Process a single video file
    """
    logging.info(f"Processing {video_path} for frames {frame_ranges}")
    return extract_frames_ffmpeg(video_path, output_dir, frame_ranges)

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
            tasks.append((video_path, frames_dir, frame_ranges[3:4]))
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
