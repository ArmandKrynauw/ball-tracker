from ultralytics import YOLO
import torch
from utils import DATA_DIR, DATASET_DIR, download_from_drive
import shutil
from pathlib import Path
import random
import yaml

FH_DATASET_DIR = DATA_DIR / "fh_dataset"

def prepare_training_data():
    """Prepare and split the dataset for training"""
    # Create necessary directories
    train_images = FH_DATASET_DIR / 'train' / 'images'
    train_labels = FH_DATASET_DIR / 'train' / 'labels'
    val_images = FH_DATASET_DIR / 'val' / 'images'
    val_labels = FH_DATASET_DIR / 'val' / 'labels'
    
    for dir in [train_images, train_labels, val_images, val_labels]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    source_images = (DATA_DIR / "field_hockey_yolo" / "images").glob("*.jpg")
    image_files = sorted(list(source_images))
    
    # Determine split (80% train, 20% val)
    random.seed(42)  # for reproducibility
    num_images = len(image_files)
    num_val = int(num_images * 0.1)
    val_indices = set(random.sample(range(num_images), num_val))
    
    # Split and copy files
    for idx, img_path in enumerate(image_files):
        # Get corresponding label path
        label_path = DATA_DIR / "field_hockey_yolo" / "labels" / f"{img_path.stem}.txt"
        
        if idx in val_indices:
            # Copy to validation set
            shutil.copy2(img_path, val_images / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, val_labels / label_path.name)
        else:
            # Copy to training set
            shutil.copy2(img_path, train_images / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, train_labels / label_path.name)
    
    # Create data.yaml
    data = {
        'path': str(FH_DATASET_DIR.absolute()),
        'train': str(train_images.relative_to(FH_DATASET_DIR)),
        'val': str(val_images.relative_to(FH_DATASET_DIR)),
        'names': {
            0: 'ball'
        }
    }
    
    yaml_path = FH_DATASET_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"\nDataset split successfully:")
    print(f"Training images: {len(list(train_images.glob('*.jpg')))}")
    print(f"Training labels: {len(list(train_labels.glob('*.txt')))}")
    print(f"Validation images: {len(list(val_images.glob('*.jpg')))}")
    print(f"Validation labels: {len(list(val_labels.glob('*.txt')))}")
    
    return yaml_path

def main():
    # Download and unzip data
    download_from_drive("11471SWR9hSO9Qe-gffQiFQytQFS2vn_E", "fh_data.zip")
    shutil.unpack_archive(DATA_DIR / "fh_data.zip", DATA_DIR)

    # Prepare dataset
    data_path = prepare_training_data()

    # Initialize model
    model = YOLO(DATA_DIR / "yolov8s.pt")

    # Determine device
    device = torch.device(
        0
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    # Train model
    model.train(
        data=data_path,
        epochs=20, 
        imgsz=1920,
        batch=8,
        device=device,
        # cache=True,
        # augment=True,
        # degrees=180,
        # fliplr=1,
        # flipud=1,
        # mixup=0.5
    )

if __name__ == "__main__":
    main()
