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
        bbox = tuple(map(int, bbox))
        self.tracking_active = self.tracker.init(frame, bbox)
        self.last_bbox = bbox if self.tracking_active else None
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

def filter_and_interpolate_predictions(predictions, image_paths, max_gap=5, max_distance=100):
    tracker = KCFTracker()
    filtered_predictions = {}
    previous_position = None
    tracking_mode = False
    last_detection_frame = -1

    for frame_index in sorted(predictions.keys()):
        frame = cv2.imread(str(image_paths[frame_index]))
        current_pred = predictions[frame_index]
        detection_available = len(current_pred['boxes']) > 0

        if detection_available:
            # Get current detection
            box = current_pred['boxes'][0]['xyxy']
            box = [float(coord) for coord in box]
            
            # Store original detection
            filtered_predictions[frame_index] = {
                'boxes': [{
                    'xyxy': np.array([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                }]
            }

            # Convert to tracker format (x, y, w, h)
            w = int(box[2] - box[0])
            h = int(box[3] - box[1])
            tracker_bbox = (int(box[0]), int(box[1]), w, h)

            if tracker.init(frame, tracker_bbox):
                tracking_mode = True
                last_detection_frame = frame_index
                previous_position = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        elif tracking_mode and frame_index - last_detection_frame <= max_gap:
            success, tracked_bbox = tracker.update(frame)
            if success:
                filtered_predictions[frame_index] = {
                    'boxes': [{
                        'xyxy': np.array([
                            tracked_bbox[0],
                            tracked_bbox[1],
                            tracked_bbox[0] + tracked_bbox[2],
                            tracked_bbox[1] + tracked_bbox[3]
                        ])
                    }]
                }
            else:
                filtered_predictions[frame_index] = {'boxes': []}
                tracking_mode = False
                tracker.reset()
        else:
            filtered_predictions[frame_index] = {'boxes': []}
            if tracking_mode:
                tracking_mode = False
                tracker.reset()

    return filtered_predictions
