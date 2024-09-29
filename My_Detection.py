import cv2
import os
import supervision as sv
from ultralytics import YOLO

def crop_center(frame, cropx, cropy):
    y, x, _ = frame.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return frame[starty:starty+cropy, startx:startx+cropx]

model = YOLO('Yejun.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Unable to read camera feed")

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cropped_frame = crop_center(frame, 640, 640)

    # frame = cropped_frame

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

    if 'homecam' in detections.data['class_name']:
        print("homecam detected!")
    else:
        print("homecam not detected.")
        
    cv2.imshow("Webcam", annotated_image)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()

