import numpy as np, os
from recognition.face_manager import FaceRecognitionManager

def test_face_manager_enroll_and_save(tmp_path):
    gallery = tmp_path / 'gallery.json'
    frm = FaceRecognitionManager(gallery_path=str(gallery))
    emb = np.random.rand(128).astype(np.float32)
    frm.enroll_person('tester', emb, face_crop=None, retrain=False)
    assert 'tester' in frm.embeddings
    frm.save_gallery()
    assert os.path.exists(str(gallery))
    frm2 = FaceRecognitionManager(gallery_path=str(gallery))
    assert 'tester' in frm2.embeddings
