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
import socket

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
gesture_threshold = 5
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
# cap = cv2.VideoCapture(0)


# Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8080))  # Listen on all interfaces
server_socket.listen()

print("Server is listening...")

def receive_image_from_client(client_socket):
    data = bytearray()
    while True:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data.extend(packet)

        # Check for JPEG end marker
        if data[-2:] == b'\xFF\xD9':
            break

    return data



def Integrated_Command(img):
    global global_current_gesture, current_device

    text = "Gesture : " + str(global_current_gesture) + ", Device : " + str(current_device)

    print(text)
    cv2.putText(img, text=text, org=(50, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

    if global_current_gesture == None or current_device == None:
        print("No gesture or device detected")
        cv2.putText(img, text="No gesture or device detected", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
        return
    elif global_current_gesture == "plug on" and current_device == "plug":
        # send_command("plug on")
        print("Plug on")
        cv2.putText(img, text="plug on", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
        return
    elif global_current_gesture == "plug off" and current_device == "plug":
        # send_command("plug off")
        print("Plug off")
        cv2.putText(img, text="plug off", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
        return 
    elif global_current_gesture == "camera on" and current_device == "homecam":
        # send_command("camera on")
        print("Camera on")
        cv2.putText(img, text="camera on", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
        return
    elif global_current_gesture == "camera off" and current_device == "homecam":
        # send_command("camera off")
        print("Camera off")
        cv2.putText(img, text="camera off", org=(50, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)
        return

def enhance_image(img):
    # Apply smoothing
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply sharpening
    img = sharpen_image(img)

    return img

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img


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

            # if is_on_camera:
            #     direction = recognize_direction(joint)
            #     if direction == current_direction:
            #         direction_count += 1
            #     else:
            #         direction_count = 1
            #         current_direction = direction

            #     if direction_count >= direction_threshold:
            #         current_gesture = "send_image_capture"
            #         global_current_gesture = "send_image_capture"
            #         # send_image_capture_command()
            #         direction_count = 0

            if gesture_count >= gesture_threshold:
                if text == "5" and not is_on:
                    current_gesture = "plug on"
                    global_current_gesture = "plug on"
                    send_command("plug on")
                    is_on = True
                    is_off = False
                elif text == "3" and not is_off:
                    current_gesture = "plug off"
                    global_current_gesture = "plug off"
                    send_command("plug off")
                    is_off = True
                    is_on = False
                elif text == "1" and not is_on_camera:
                    current_gesture = "camera on"
                    global_current_gesture = "camera on"
                    send_command("camera on")
                    is_on_camera = True
                    is_off_camera = False
                elif text == "0" and not is_off_camera:
                    current_gesture = "camera off"
                    global_current_gesture = "camera off"
                    send_command("camera off")
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

def preprocess_image(img):
    # Example usage after receiving the image:
    # img = rotate_image(img, 90)  # Rotate 90 degrees clockwise

    # Step 1: Upscale the image
    img = upscale_image(img)

    # Step 2: Denoise the image
    img = denoise_image(img)

    # Step 3: Adjust brightness and contrast
    # img = adjust_brightness_contrast(img)

    # Step 4: Sharpen the image
    # img = sharpen_image(img)

    # Step 5: Optionally, apply histogram equalization
    # img = equalize_histogram(img)

    return img


def upscale_image(img):
    # Define the scale factor
    scale_percent = 200  # Upscale by 200%
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    upscaled_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return upscaled_img


def denoise_image(img):
    # Resize image to smaller dimensions
    img_small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # Apply denoising
    denoised_small = cv2.fastNlMeansDenoisingColored(
        img_small, None, h=3, hColor=3, templateWindowSize=3, searchWindowSize=7
    )
    
    # Resize back to original dimensions
    denoised_img = cv2.resize(denoised_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return denoised_img

def rotate_image(img, angle):
    if angle == 90:
        # Rotate the image 90 degrees clockwise
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        # Rotate the image 90 degrees counterclockwise
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Default: no rotation
        return img


def adjust_brightness_contrast(img, brightness=0, contrast=30):
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img

def sharpen_image(img):
    # Create a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    # Apply the kernel to the image
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img

def equalize_histogram(img):
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(img_y_cr_cb)

    # Apply histogram equalization to the Y channel
    y_eq = cv2.equalizeHist(y_channel)

    # Merge the channels back
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))

    # Convert back to BGR color space
    img_output = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)
    return img_output


try:
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established.")
        
        try:
            while client_socket:
                img_data = receive_image_from_client(client_socket)

                if img_data:
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        # Convert image to RGB for MediaPipe
                        # img = enhance_image(img)

                        img = preprocess_image(img)


                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        hands_result = hands.process(img_rgb)

                        # Create threads for gesture recognition and object detection
                        gesture_thread = threading.Thread(target=process_gesture, args=(img, hands_result))
                        detection_thread = threading.Thread(target=process_object_detection, args=(img,))

                        # Start threads
                        gesture_thread.start()
                        detection_thread.start()

                        # Wait for both threads to finish
                        gesture_thread.join()
                        detection_thread.join()

                        # Integrated command execution
                        Integrated_Command(img)

                        # Display the result
                        cv2.imshow("Combined Detection", img)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Failed to decode image")
                else:
                    # No more data from client
                    break

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
            print(f"Connection from {client_address} was lost: {e}")
            cv2.destroyAllWindows()

        finally:
            cv2.destroyAllWindows()
            client_socket.close()
            print(f"Connection with {client_address} closed.")

except KeyboardInterrupt:
    print("Server shutting down...")
finally:
    server_socket.close()
    cv2.destroyAllWindows()

