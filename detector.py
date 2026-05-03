import os
import logging
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Detects faces in a given image using a YOLO model.
    """
    def __init__(self, model_path: str = "models/yolov8n-face.pt"):
        """
        Initialize the FaceDetector with the given model path.
        Raises FileNotFoundError if the model file does not exist.
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in a frame.

        Args:
            frame (np.ndarray): Input image/frame.

        Returns:
            List of dicts with coordinates and confidence for each detected face.
            Example: [{'box': (x1, y1, x2, y2), 'confidence': score}, ...]
        """
        faces = []
        try:
            results = self.model(frame, verbose=False)
            for r in results:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else [None] * len(xyxy)
                for box, score in zip(xyxy, conf):
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append({'box': (x1, y1, x2, y2), 'confidence': float(score) if score is not None else None})
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        return faces
