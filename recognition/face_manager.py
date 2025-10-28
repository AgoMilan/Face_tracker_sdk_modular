# recognition/face_manager.py
import os, json, cv2, numpy as np, warnings, torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")
USE_INSIGHTFACE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    USE_INSIGHTFACE = True
except Exception:
    import face_recognition
class FaceRecognitionManager:
    def __init__(self, gallery_path="gallery.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = {}
        self.gallery_path = gallery_path
        self.pca = None
        self.face_dir = "thumbnails"
        os.makedirs(self.face_dir, exist_ok=True)
        if USE_INSIGHTFACE:
            try:
                self.model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
            except Exception:
                self.model = None
        else:
            self.model = None
        self.load_gallery()
    def save_gallery(self):
        data = {k: [v.tolist() for v in val] for k, val in self.embeddings.items()}
        with open(self.gallery_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    def load_gallery(self):
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                self.embeddings = {}
                return
            clean = {}
            for k, v in data.items():
                arrs = []
                for e in v:
                    try:
                        e_np = np.array(e, dtype=np.float32)
                        if e_np.ndim == 1 and e_np.shape[0] >= 100:
                            arrs.append(e_np)
                    except Exception:
                        pass
                if arrs:
                    clean[k] = arrs
            self.embeddings = clean
        else:
            self.embeddings = {}
    def crop_face(self, frame, bbox, scale=3.0):
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return None
        h, w = frame.shape[:2]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = x1 + box_w // 2, y1 + box_h // 2
        new_w, new_h = int(box_w * scale), int(box_h * scale)
        x1n, y1n = int(cx - new_w // 2), int(cy - new_h // 2)
        x2n, y2n = int(cx + new_w // 2), int(cy + new_h // 2)
        x1n, y1n, x2n, y2n = max(0, x1n), max(0, y1n), min(w, x2n), min(h, y2n)
        if x2n - x1n <= 0 or y2n - y1n <= 0:
            return None
        return frame[y1n:y2n, x1n:x2n].copy()
    def get_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None
        if USE_INSIGHTFACE and self.model is not None:
            try:
                faces = self.model.get(face_crop)
                if not faces:
                    return None
                emb = faces[0].normed_embedding
            except Exception:
                return None
        else:
            try:
                face_enc = face_recognition.face_encodings(face_crop)
                if not face_enc:
                    return None
                emb = face_enc[0]
            except Exception:
                return None
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[0] < 100:
            return None
        emb = normalize(emb.reshape(1, -1))[0]
        if self.pca is not None:
            try:
                emb = self.pca.transform([emb])[0]
            except Exception:
                pass
        return emb
    def enroll_person(self, name, emb, face_crop=None, retrain=True):
        if emb is None:
            return
        if name not in self.embeddings:
            self.embeddings[name] = []
        self.embeddings[name].append(emb)
        if face_crop is not None:
            try:
                cv2.imwrite(os.path.join(self.face_dir, f"{name}.jpg"), face_crop)
            except Exception:
                pass
        self.save_gallery()
        if retrain:
            self._retrain_pca()
    def add_embedding_to_person(self, name, frame, bbox):
        if name not in self.embeddings:
            return False
        face_crop = self.crop_face(frame, bbox, scale=3)
        emb = self.get_embedding(face_crop)
        if emb is None:
            return False
        self.enroll_person(name, emb, face_crop=face_crop, retrain=False)
        return True
    def clean_gallery(self, keep_best=10):
        cleaned = 0
        for name, embs in list(self.embeddings.items()):
            if len(embs) <= keep_best:
                continue
            sims = []
            for i, e1 in enumerate(embs):
                s = np.mean([np.dot(e1, e2) for j, e2 in enumerate(embs) if i != j]) if len(embs) > 1 else 0.0
                sims.append((s, e1))
            sims.sort(key=lambda x: x[0], reverse=True)
            self.embeddings[name] = [e for _, e in sims[:keep_best]]
            cleaned += 1
        if cleaned:
            self.save_gallery()
            self._retrain_pca()
    def _retrain_pca(self):
        all_embs = [e for v in self.embeddings.values() for e in v if e is not None]
        if len(all_embs) < 3:
            return
        lengths = [e.shape[0] for e in all_embs]
        main_len = max(set(lengths), key=lengths.count)
        compatible = [e for e in all_embs if e.shape[0] == main_len]
        if len(compatible) < 3:
            return
        data = np.vstack(compatible)
        n_components = min(64, data.shape[1], len(compatible) - 1)
        if n_components < 2:
            return
        try:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(data)
        except Exception:
            pass
    def _match_face(self, emb):
        if emb is None or not self.embeddings:
            return "unknown", 0.0
        best_name, best_sim = "unknown", 0.0
        for name, embs in self.embeddings.items():
            sims = [np.dot(emb, e) for e in embs if e.shape == emb.shape]
            if sims:
                sim = float(np.mean(sims))
                if sim > best_sim:
                    best_sim, best_name = sim, name
        threshold = 0.43 if USE_INSIGHTFACE else 0.58
        return (best_name, best_sim) if best_sim > threshold else ("unknown", best_sim)
