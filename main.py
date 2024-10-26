import logging
import shutil

import torch
from ultralytics import YOLO
from tqdm.notebook import tqdm
import pandas as pd
import os
from pathlib import Path


from utils import (
    DATA_DIR,
    copy_roboflow_data,
    create_yolo_data_file,
    download_from_drive,
)

logger = logging.getLogger(__name__)

download_from_drive("1XGD6ZRUiFVuvpKsfOQLreA27lTRM3d2-", "images.zip")


if not (DATA_DIR / "images").exists():
    if not (DATA_DIR / "images.zip").exists():
        raise FileNotFoundError(f"images.zip file not found in {DATA_DIR}")

    logger.info("Unzipping images.zip...")
    shutil.unpack_archive(DATA_DIR / "images.zip", DATA_DIR / "images")

copy_roboflow_data(DATA_DIR / "images")

data_path = create_yolo_data_file(["puck"]) 

# Determine the device to use
device = torch.device(
    0
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = YOLO(DATA_DIR / "yolo11n.pt")

detect  = Path(os.getcwd()) / "runs/detect"


results = model.train(
    data=data_path,
    epochs=30,
    batch=4,
    imgsz=640,
    device=device,
    # fraction=0.3,
    cache=True,
    project=detect,
    name="run",
    max_det=1,
    translate=0.3,
    optimizer="RAdam",
)


# Grab save dir from dictionary
best_weights_path = results.save_dir / 'weights/best.pt'
last_weights_path = results.save_dir / 'weights/last.pt'   


# Load model for validation
model = YOLO(best_weights_path)


print('\nPerforming testing...\n')


results = model.val(
    data=data_path,
    project=results.save_dir,
    name="test",
    split='test', # Use the test images in yaml
    max_det=1,
)

print('Testing completed.') 