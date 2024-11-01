from pathlib import Path
import subprocess
import pandas as pd
from joblib import Parallel, delayed

DATA = Path("data/field_hockey")

videos_path = DATA / "videos"
clips_path = DATA / "annotations" / "clips.csv"

def export_clip(video_path, start_frame, end_frame):
    if not video_path:
        return None

    original_filename = Path(video_path).stem
    output_filename = f"{original_filename}_{start_frame}-{end_frame}.mp4"
    output_path = Path(video_path).parent / output_filename

    if output_path.exists():
        print(f"Clip {output_filename} already exists, skipping...")
        return output_path

    command = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-an',
        str(output_path)
    ]

    try:
        print(f"Exporting clip {output_filename}...")
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print("Export Error", f"Failed to export clip: {str(e)}")
        return None

def main():
    clips = pd.read_csv(clips_path)

    # Be warned. This will kill your CPU lol. Couldn't bother to implement the
    # clipping with ffmpeg using a gpu codecs
    Parallel(n_jobs=-1)(
        delayed(export_clip)(
            videos_path / f"{clip['id']}.mp4",
            clip['start_frame'],
            clip['end_frame']
        ) for i, clip in clips.iterrows()
    )

if __name__ == "__main__":
    main()