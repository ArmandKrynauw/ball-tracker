import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.distance import euclidean
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KCFTracker:
    def __init__(self):
        self.tracker = None
        self.last_bbox = None
        self.tracking_active = False

    def init(self, frame, bbox):
        self.tracker = cv2.TrackerKCF_create()
        x, y = int(bbox[0]), int(bbox[1])
        w = max(1, int(bbox[2] - bbox[0]))
        h = max(1, int(bbox[3] - bbox[1]))
        bbox_tuple = (x, y, w, h)
        self.tracking_active = self.tracker.init(frame, bbox_tuple)
        self.last_bbox = bbox_tuple if self.tracking_active else None
        return self.tracking_active

    def update(self, frame):
        if not self.tracking_active:
            return False, None
        success, bbox = self.tracker.update(frame)
        if success:
            self.last_bbox = tuple(map(int, bbox))
        return success, self.last_bbox

    def reset(self):
        self.tracker = None
        self.last_bbox = None
        self.tracking_active = False

def get_box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def filter_and_interpolate_predictions(predictions, image_paths, max_gap=5):
    filtered_predictions = {}
    
    # First, find all frames with valid detections
    valid_frames = []
    valid_boxes = []
    for frame_idx in sorted(predictions.keys()):
        if len(predictions[frame_idx]['boxes']) > 0:
            valid_frames.append(frame_idx)
            valid_boxes.append(predictions[frame_idx]['boxes'][0]['xyxy'])
    
    # Process each pair of consecutive valid detections
    for i in range(len(valid_frames) - 1):
        start_frame = valid_frames[i]
        end_frame = valid_frames[i + 1]
        frame_gap = end_frame - start_frame
        
        # If frames are consecutive or gap is too large, just keep original detections
        if frame_gap <= 1 or frame_gap > max_gap:
            filtered_predictions[start_frame] = predictions[start_frame]
            continue
        
        # Get start and end positions
        start_box = valid_boxes[i]
        end_box = valid_boxes[i + 1]
        
        # Store the start frame detection
        filtered_predictions[start_frame] = predictions[start_frame]
        
        # Interpolate intermediate frames
        for frame_idx in range(start_frame + 1, end_frame):
            # Calculate interpolation factor
            t = (frame_idx - start_frame) / frame_gap
            
            # Interpolate box coordinates
            interpolated_box = []
            for j in range(4):  # For each coordinate (x1, y1, x2, y2)
                interp_coord = start_box[j] + t * (end_box[j] - start_box[j])
                interpolated_box.append(interp_coord)
            
            # Store interpolated detection
            filtered_predictions[frame_idx] = {
                'boxes': [{
                    'xyxy': np.array(interpolated_box)
                }]
            }
    
    # Add the last valid frame
    if valid_frames:
        filtered_predictions[valid_frames[-1]] = predictions[valid_frames[-1]]
    
    # Fill in any remaining frames with empty boxes
    for frame_idx in predictions.keys():
        if frame_idx not in filtered_predictions:
            filtered_predictions[frame_idx] = {'boxes': []}
    
    return filtered_predictions