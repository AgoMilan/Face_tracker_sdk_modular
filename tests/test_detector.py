import numpy as np
from detection.face_detector import FaceDetector

def test_detector_init_and_track(monkeypatch):
    class DummyModel:
        def __init__(self, path): self.path = path
        def track(self, frame, persist, verbose, classes): return []
        def __call__(self, frame): return []
    import detection.face_detector as mod
    monkeypatch.setattr(mod, 'YOLO', DummyModel)
    det = FaceDetector(model_path='models/YOLO11face.pt')
    img = (np.ones((64,64,3))*255).astype('uint8')
    res = det.track(img)
    assert isinstance(res, list)
