from pathlib import Path
import shutil
import yaml

def prepare_yolo_dataset(base_dir, output_dir):
    """
    Reorganize the data into flat YOLOv8 format:
    
    output_dir/
        images/
            fh_01_991_frame_0001.jpg
            ...
        labels/
            fh_01_991_frame_0001.txt
            ...
        data.yaml
    """
    # Set up directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Source directories
    frames_dir = base_dir / 'frames'
    annotations_dir = base_dir / 'yolo_annotations'
    
    print("Copying files...")
    
    # Process each clip directory
    for clip_dir in frames_dir.glob("*"):
        clip_name = clip_dir.name  # e.g., "fh_01_991-1215"
        
        # Copy frames
        for frame_path in clip_dir.glob("frame_*.jpg"):
            new_name = f"{clip_name}_{frame_path.name}"
            shutil.copy2(frame_path, images_dir / new_name)
        
        # Copy corresponding annotations
        anno_dir = annotations_dir / clip_name
        if anno_dir.exists():
            for label_path in anno_dir.glob("frame_*.txt"):
                new_name = f"{clip_name}_{label_path.name}"
                shutil.copy2(label_path, labels_dir / new_name)
    
    # Create data.yaml
    data = {
        'path': str(output_dir.absolute()),
        'train': 'images',  # Just point to the images directory
        'val': 'images',    # Can be updated later after splitting
        'names': {
            0: 'ball'
        }
    }
    
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    # Print summary
    num_images = len(list(images_dir.glob('*.jpg')))
    num_labels = len(list(labels_dir.glob('*.txt')))
    print("\nDataset prepared successfully:")
    print(f"Images: {num_images}")
    print(f"Labels: {num_labels}")
    print(f"\nData YAML file created at: {output_dir / 'data.yaml'}")

def main():
    base_dir = Path('data/field_hockey')
    output_dir = Path('data/field_hockey_yolo')
    
    prepare_yolo_dataset(base_dir, output_dir)

if __name__ == '__main__':
    main()