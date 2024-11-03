import logging
import shutil

import torch
from ultralytics import YOLO
from tqdm.notebook import tqdm
import pandas as pd
import os
from pathlib import Path
from tools.relabel import main as relabel
from tools.pipe_images import main as pipe_images

from utils import (
    DATA_DIR,
    setup_dataset,
    create_yolo_data_file,
    download_from_drive,
    unzipData,
)

logger = logging.getLogger(__name__)

# Download and extraction of data
download_from_drive("1XGD6ZRUiFVuvpKsfOQLreA27lTRM3d2-",DATA_DIR, "images.zip")
download_from_drive("1XGD6ZRUiFVuvpKsfOQLreA27lTRM3d2-",DATA_DIR, "annotations.zip")

# download_from_drive('1XjdCJAxRXw7r2vkbGitc3zh1pxmSU9D_',DATA_DIR / "field_hockey","field_hockey.zip")


unzipData(DATA_DIR / "images","images.zip")
unzipData(DATA_DIR / "annotations","annotations.zip")
# Amendments to dataset

setup_dataset()
relabel("data/dataset/labels")

# Run pipeline on data (Pre-Processing)

pipe_images()


data_path = create_yolo_data_file(["ball","puck"]) 

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
    imgsz=1280,
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
# model = YOLO(best_weights_path)


# print('\nPerforming testing...\n')


# results = model.val(
#     data=data_path,
#     project=results.save_dir,
#     name="test",
#     split='test', # Use the test images in yaml
#     max_det=1,
# )

# print('Testing completed.') 