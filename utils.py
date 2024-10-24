import logging
import os
import shutil
from pathlib import Path
from typing import List

import gdown
import tqdm
import yaml

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "dataset"

# Images directories
IMAGES_DIR = DATASET_DIR / "images"
TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VALID_IMAGES_DIR = IMAGES_DIR / "valid"
TEST_IMAGES_DIR = IMAGES_DIR / "test"

# Labels directories
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_LABELS_DIR = LABELS_DIR / "train"
VALID_LABELS_DIR = LABELS_DIR / "valid"
TEST_LABELS_DIR = LABELS_DIR / "test"

YOLO_DATASET_DIRS = [
    TRAIN_IMAGES_DIR,
    VALID_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VALID_LABELS_DIR,
    TEST_LABELS_DIR,
]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler to the logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def download_from_drive(id, name: str):
    output_path = DATA_DIR / name

    if output_path.exists():
        logger.info(f"{name} already exists in {output_path}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {name}...")
    gdown.download(
        f"https://drive.google.com/uc?id={id}", str(output_path), quiet=False
    )

    return output_path


def copy_roboflow_data(data_dir: Path, output_data_dir: Path | None = None):
    if not output_data_dir:
        output_data_dir = DATASET_DIR

    for dir in YOLO_DATASET_DIRS:
        dir.mkdir(parents=True, exist_ok=True)

    train_images_path = data_dir / "train" / "images"
    for img in tqdm.tqdm(train_images_path.glob("*"), desc="Copying training images"):
        shutil.copy(img, TRAIN_IMAGES_DIR / img.name)

    val_images_path = data_dir / "valid" / "images"
    for img in tqdm.tqdm(val_images_path.glob("*"), desc="Copying validation images"):
        shutil.copy(img, VALID_IMAGES_DIR / img.name)

    test_images_path = data_dir / "test" / "images"
    for img in tqdm.tqdm(test_images_path.glob("*"), desc="Copying test images"):
        shutil.copy(img, TEST_IMAGES_DIR / img.name)

    train_labels_path = data_dir / "train" / "labels"
    for label in tqdm.tqdm(train_labels_path.glob("*"), desc="Copying training labels"):
        shutil.copy(label, TRAIN_LABELS_DIR / label.name)

    val_labels_path = data_dir / "valid" / "labels"
    for label in tqdm.tqdm(val_labels_path.glob("*"), desc="Copying validation labels"):
        shutil.copy(label, VALID_LABELS_DIR / label.name)

    test_labels_path = data_dir / "test" / "labels"
    for label in tqdm.tqdm(test_labels_path.glob("*"), desc="Copying test labels"):
        shutil.copy(label, TEST_LABELS_DIR / label.name)


def create_yolo_data_file(class_names: List[str]):
    cwd = Path(os.getcwd())

    data_yaml = {
        "train": str(cwd / TRAIN_IMAGES_DIR),
        "val": str(cwd / VALID_IMAGES_DIR),
        "test": str(cwd / TEST_IMAGES_DIR),
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = DATA_DIR / "data.yaml"
    with open(yaml_path, "w") as file:
        yaml.dump(data_yaml, file, default_flow_style=False)

    return yaml_path
