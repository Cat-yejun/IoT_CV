import cv2
import mediapipe as mp
import numpy as np
import threading
import supervision as sv
from ultralytics import YOLO
from Gesture_Recognition import load_gesture_model, recognize_gesture, recognize_direction
from Configuration import max_num_hands
from Smart_Things import send_command, send_camera_move_command, send_image_capture_command
import string

# Gesture recognition setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

knn = load_gesture_model()

# YOLO setup
model = YOLO('Yejun.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
current_device = None

# Initialize variables for gesture recognition
sentence = ""
last_gesture = ""
current_gesture = None
gesture_count = 0
gesture_threshold = 10
completed_sentences = []
is_on = False
is_off = False
is_on_camera = False
is_off_camera = False
global_current_gesture = None

# Variables for direction tracking
current_direction = ""
direction_count = 0
direction_threshold = 10

# Initialize video capture
cap = cv2.VideoCapture(0)


def Integrated_Command(img):
    global global_current_gesture, current_device

    text = "Gesture : " + str(global_current_gesture) + ", Device : " + str(current_device)

    print(text)
    cv2.putText(img, text=text, org=(50, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)

    if global_current_gesture == None or current_device == None:
        print("No gesture or device detected")
        cv2.putText(img, text="No gesture or device detected", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
        return
    elif global_current_gesture == "plug on" and current_device == "plug":
        # send_command("plug on")
        print("Plug on")
        cv2.putText(img, text="plug on", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
        return
    elif global_current_gesture == "plug off" and current_device == "plug":
        # send_command("plug off")
        print("Plug off")
        cv2.putText(img, text="plug off", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
        return 
    elif global_current_gesture == "camera on" and current_device == "homecam":
        # send_command("camera on")
        print("Camera on")
        cv2.putText(img, text="camera on", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
        return
    elif global_current_gesture == "camera off" and current_device == "homecam":
        # send_command("camera off")
        print("Camera off")
        cv2.putText(img, text="camera off", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=3)
        return



def process_gesture(img, hands_result):
    global gesture_count, current_gesture, direction_count, current_direction, is_on, is_off, is_on_camera, is_off_camera, global_current_gesture

    hand_results = []

    if hands_result.multi_hand_landmarks is not None:
        for res in hands_result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            gesture_result = recognize_gesture(knn, joint)
            if gesture_result:
                hand_results.append(gesture_result)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if len(hand_results) == 1:
            text = hand_results[0]

            if text == current_gesture:
                gesture_count += 1
            else:
                gesture_count = 1
                current_gesture = text

            if is_on_camera:
                direction = recognize_direction(joint)
                if direction == current_direction:
                    direction_count += 1
                else:
                    direction_count = 1
                    current_direction = direction

                if direction_count >= direction_threshold:
                    current_gesture = "send_image_capture"
                    global_current_gesture = "send_image_capture"
                    # send_image_capture_command()
                    direction_count = 0

            if gesture_count >= gesture_threshold:
                if text == "5" and not is_on:
                    current_gesture = "plug on"
                    global_current_gesture = "plug on"
                    # send_command("plug on")
                    is_on = True
                    is_off = False
                elif text == "0" and not is_off:
                    current_gesture = "plug off"
                    global_current_gesture = "plug off"
                    # send_command("plug off")
                    is_off = True
                    is_on = False
                elif text == "1" and not is_on_camera:
                    current_gesture = "camera on"
                    global_current_gesture = "camera on"
                    # send_command("camera on")
                    is_on_camera = True
                    is_off_camera = False
                elif text == "3" and not is_off_camera:
                    current_gesture = "camera off"
                    global_current_gesture = "camera off"
                    # send_command("camera off")
                    is_off_camera = True
                    is_on_camera = False

                gesture_count = 0

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
            x, y = 10, 30
            cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
            cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)



def process_object_detection(img):
    global current_device

    results = model(img)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    annotated_image = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    if 'homecam' in detections.data['class_name']:
        current_device = "homecam"
    else:
        current_device = None
        
    return annotated_image


if __name__ == "__main__":

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process gesture recognition
        hands_result = hands.process(frame)

        # Convert back to BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Thread for gesture processing
        gesture_thread = threading.Thread(target=process_gesture, args=(frame, hands_result))
        gesture_thread.start()

        # Thread for object detection processing
        detection_thread = threading.Thread(target=process_object_detection, args=(frame,))
        detection_thread.start()

        # Wait for both threads to finish
        gesture_thread.join()
        detection_thread.join()

        Integrated_Command(frame)

        # Show result
        cv2.imshow("Combined Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
