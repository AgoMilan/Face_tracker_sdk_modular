# utils/draw.py
import cv2
def draw_label(frame, bbox, label):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
