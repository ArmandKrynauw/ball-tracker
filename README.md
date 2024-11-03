# Fine-Tuning Yolo on the Field Hockey Dataset

Run the `train_fh.py` script to fine-tune YOLO on the field hockey dataset.
This script will download the required data from Google Drive and train YOLO
on the field hockey dataset.

# Field Hockey Data

All the data for the field hockey dataset can be downloaded from this
[link](https://drive.google.com/file/d/1XBnIeDy4JmkCozLOqJVY2KkWFLxyHGct/view?usp=drive_link).

## Data Structure

For the primary field hockey dataset, the data is structured as follows:

```
data/field_hockey
├── annotations
│   ├── clips.csv
│   ├── map.csv
│   ├── fh_01_1344-1531.json
│   ├── ...
│   └── fh_01_1733-1982.json
└── videos
    ├── fh_01.mp4
    ├── fh_02.mp4
    └── fh_03.mp4
```

The `annotations` folder contains the following files:
- `clips.csv`: A CSV file containing the start and end times of each clip in the
  dataset.
- `map.csv`: A CSV file containing the description of each video in the dataset.
- `fh_01_1344-1531.json`: A JSON file containing the annotations for a specific
  clip in the dataset from `Label Studio`.

The `videos` folder contains the video files in the dataset.

## Processing the Data

There are a couple of Python scripts that can be used process the data from the
original video files and annotations.

### Extracting Clips

To extract clips from the original video files, you can use `clips.py`. This
script takes the `clips.csv` file as input and extracts the clips from the
original video files, such as `fh_01.mp4`.

### Converting Annotations

To convert the annotations from `Label Studio` to a format that can be used by
YOLO, you can use `converter.py`. This script takes the `clips.csv` file and the
the JSON files from `Label Studio` as input and converts the annotations. If a
corresponding annotation file is not found for a clip, the script will just
output empty annotations for it.

### Splitting the Clips into Frames

To split the clips into frames, you can use `splitter.py`. This script takes the
`clips.csv` and the JSON files from `Label Studio` as input and splits the
original video files into frames according the the clips. If a corresponding
annotation file is not found for a clip, the script will still split the clip
into frames.


### Visualizing the Data

To perform a quick sanity check on the annotations, you can use `visualize.py`.
This will use the folder structure created by `converter.py` and `splitter.py`
to visualize the annotations on the frames grouped by clips.

### Flattening the Annotations and Frames to be used by YOLO

To flatten the annotations and frames to be used by YOLO, you can use
`prepare_yolo_dataset.py`. 
