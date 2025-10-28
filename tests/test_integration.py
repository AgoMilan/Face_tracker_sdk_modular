import numpy as np
import types
from detection.face_detector import FaceDetector
from recognition.face_manager import FaceRecognitionManager

def test_integration_detector_and_frm(monkeypatch):
    class BoxesObj:
        def __init__(self):
            import numpy as _np, torch as _t
            self.xyxy = _np.array([[10,10,40,40]], dtype=float)
            self.id = _t.tensor([0])
    class DummyModel:
        def __init__(self, path): pass
        def track(self, frame, persist, verbose, classes):
            return [types.SimpleNamespace(boxes=BoxesObj())]
    import detection.face_detector as mod
    monkeypatch.setattr(mod, 'YOLO', DummyModel)
    det = FaceDetector('models/YOLO11face.pt')
    frm = FaceRecognitionManager(gallery_path=':memory:')
    frame = (np.ones((100,100,3))*255).astype('uint8')
    res = det.track(frame)
    assert isinstance(res, list)
