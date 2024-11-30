import os.path
import time
import numpy as np
from ultralytics import YOLO
import cv2
import folder_paths
from huggingface_hub import hf_hub_download

def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    draw_img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
    return draw_img


class Detector:
    def __init__(self):
        model_path = os.path.join(folder_paths.models_dir, "sapiens/yolov8m.pt")
        if not os.path.exists(model_path):
            print(f"No yolo pt in sapiens dir,auto download from 'Ultralytics/YOLOv8'")
            hf_hub_download(repo_id="Ultralytics/YOLOv8", filename="yolov8m.pt", local_dir=os.path.join(folder_paths.models_dir, "sapiens"))
        self.model = YOLO(model_path)
        self.person_id = 0
        self.conf_thres = 0.25

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        results = self.model(img, conf=self.conf_thres)
        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

        # Filter out only person
        person_detections = detections[detections[:, -1] == self.person_id]
        boxes = person_detections[:, :-2].astype(int)

        #print(f"Detection inference took: {time.perf_counter() - start:.4f} seconds")
        return boxes


