# main.py - hlavní smyčka (modulární)
import time, cv2
from detection.face_detector import FaceDetector
from recognition.face_manager import FaceRecognitionManager
from camera.canon_sdk import CanonCamera
from utils.config import YOLO_MODEL, GALLERY_FILE
from utils.draw import draw_label

def main():
    print("[MAIN] Načítám YOLO model...")
    detector = FaceDetector(YOLO_MODEL)
    frm = FaceRecognitionManager(GALLERY_FILE)
    cam = CanonCamera()
    try:
        cam.init()
        cam.start_liveview()
    except Exception as e:
        print(f"[MAIN] Warning: camera init/start failed: {e} (continuing, get_frame may return None)")

    print("Ovládání: [r] přejmenuj | [a] přidej embedding | [c] čistit galerii | [ESC] konec")
    scale = 2
    current_label = None
    current_bbox = None
    gallery_dirty = False

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_time))
            prev_time = now

            # detect + track
            results = detector.track(frame)

            if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
                boxes = getattr(results[0].boxes, "xyxy", None)
                ids_attr = getattr(results[0].boxes, "id", None)
                boxes_np = boxes.cpu().numpy() if boxes is not None else []
                ids_np = ids_attr.cpu().numpy().astype(int) if ids_attr is not None else [None] * len(boxes_np)

                for bbox, tid in zip(boxes_np, ids_np):
                    face_crop = frm.crop_face(frame, bbox, scale=3)
                    emb = frm.get_embedding(face_crop)
                    label, score = frm._match_face(emb)
                    current_label = label
                    current_bbox = bbox
                    x1, y1, x2, y2 = map(int, bbox)

                    emb_count = len(frm.embeddings.get(label, [])) if label != "unknown" else 0
                    display_label = f"{label} ({emb_count})" if label != "unknown" else "unknown"

                    draw_label(frame, (x1, y1, x2, y2), display_label)

            # FPS overlay top-left
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Canon Face Tracker", cv2.resize(frame, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF

            # rename (r)
            if key == ord('r') and current_bbox is not None:
                try:
                    new_name = input(f"[RENAME] Aktuální osoba '{current_label}' → nové jméno: ").strip()
                except Exception:
                    new_name = ""
                if new_name:
                    emb = frm.get_embedding(frm.crop_face(frame, current_bbox, scale=3))
                    frm.enroll_person(new_name, emb, frm.crop_face(frame, current_bbox, scale=3), retrain=False)
                    current_label = new_name
                    gallery_dirty = True
                    print(f"[LIVE] '{new_name}' přidán (PCA odložen).")

            # add embedding (a)
            elif key == ord('a') and current_label and current_label != "unknown" and current_bbox is not None:
                ok = frm.add_embedding_to_person(current_label, frame, current_bbox)
                if ok:
                    gallery_dirty = True
                    print(f"[LIVE] Přidán nový embedding pro '{current_label}' (PCA odloženo).")
                else:
                    print("[LIVE] Přidání embeddingu selhalo.")
            # clean (c)
            elif key == ord('c'):
                frm.clean_gallery(keep_best=10)
                gallery_dirty = False
            elif key == 27:
                break
    finally:
        if gallery_dirty:
            print("[FRM] ♻️ Galerie změněna během běhu — aktualizuji PCA a ukládám...")
            frm._retrain_pca()
            frm.save_gallery()
        try:
            cam.stop_liveview()
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
