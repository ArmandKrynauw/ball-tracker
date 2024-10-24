import logging
import shutil

import torch
from ultralytics import YOLO

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

results = model.train(
    data=data_path,
    epochs=30,
    batch=64,
    imgsz=640,
    device=device,
    # fraction=0.3,
    cache=True,
)
