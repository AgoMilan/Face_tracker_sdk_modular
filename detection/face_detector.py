# detection/face_detector.py
from ultralytics import YOLO
class FaceDetector:
    def __init__(self, model_path="models/YOLO11face.pt"):
        self.model = YOLO(model_path)
    def track(self, frame):
        try:
            return self.model.track(frame, persist=True, verbose=False, classes=[0])
        except Exception:
            return self.model(frame)
