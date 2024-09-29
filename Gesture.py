#gesture_use
import cv2
import mediapipe as mp
import numpy as np
from Gesture_Recognition import load_gesture_model, recognize_gesture, recognize_direction
from Configuration import max_num_hands
from Smart_Things import send_command, send_camera_move_command, send_image_capture_command

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

knn = load_gesture_model()

# Initialize variables
sentence = ""
last_gesture = ""
current_gesture = ""
gesture_count = 0
gesture_threshold = 10  # Threshold to recognize gesture after being held for 10 frames
completed_sentences = []
is_on = False
is_off = False
is_on_camera = False
is_off_camera = False

# Variables for direction tracking
current_direction = ""
direction_count = 0
direction_threshold = 10  # Threshold to recognize direction after being held for 10 frames

def use_gestures():
    global sentence, last_gesture, current_gesture, gesture_count, is_on, is_off, is_on_camera, is_off_camera
    global current_direction, direction_count

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        expanded_img = np.zeros((img.shape[0], img.shape[1] + 300, 3), dtype=np.uint8)
        expanded_img[:, :img.shape[1]] = img

        hand_results = []

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                gesture_result = recognize_gesture(knn, joint)
                if gesture_result:
                    hand_results.append(gesture_result)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(hand_results) == 1:
                text = hand_results[0]

                # Check if the same gesture is detected consecutively
                if text == current_gesture:
                    gesture_count += 1  # Increase count if the same gesture continues
                else:
                    gesture_count = 1  # Reset count if a new gesture is detected
                    current_gesture = text  # Update current gesture

                if is_on_camera == True:
                    # Detect direction and handle consecutive frames
                    direction = recognize_direction(joint)
                    if direction == current_direction:
                        direction_count += 1  # Increase count if the same direction continues
                    else:
                        direction_count = 1  # Reset count if a new direction is detected
                        current_direction = direction  # Update current direction

                    if direction_count >= direction_threshold:
                        # send_camera_move_command(direction)
                        send_image_capture_command()
                        print(current_direction)
                        direction_count = 0  # Reset after handling direction

                # Check if gesture is held for the required number of frames
                if gesture_count >= gesture_threshold:
                    if text == "5" and not is_on:
                        send_command("plug on")
                        is_on = True
                        is_off = False
                    elif text == "0" and not is_off:
                        send_command("plug off")
                        is_off = True
                        is_on = False
                    elif text == "1" and not is_on_camera:
                        send_command("camera on")
                        is_on_camera = True
                        is_off_camera = False
                    elif text == "3" and not is_off_camera:
                        send_command("camera off")
                        is_off_camera = True
                        is_on_camera = False

                    gesture_count = 0  # Reset gesture count after processing

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
                x, y = 10, 30
                cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
                cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

            elif len(hand_results) == 2:
                combined_text = hand_results[1] + hand_results[0]
                temp = int(combined_text)
                if temp < len(completed_sentences):
                    text = completed_sentences[temp]
                else:
                    text = "?"

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
                x, y = 10, 30
                cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
                cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

        expanded_img[:, :img.shape[1]] = img

        for i, completed_sentence in enumerate(completed_sentences):
            cv2.putText(expanded_img, str(i) + " : " + completed_sentence, (img.shape[1] + 10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Alphabet Recognition', expanded_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()