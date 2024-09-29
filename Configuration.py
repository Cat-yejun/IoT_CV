#config.py
import cv2

# Define the number of hands and gestures
max_num_hands = 2

gesture = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
}

# Define the gesture to alphabet mapping
alphabet_gesture = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F',
    7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
    13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
    19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X',
    25: 'Y', 26: 'Z'
}

def create_mapping_text():
    mapping_text = "Number to Alphabet Mapping:\n"
    for number, alphabet in alphabet_gesture.items():
        mapping_text += f"{number}: {alphabet}    "
    return mapping_text

def overlay_mapping_text(img, text, start_x=10, start_y=30, line_height=30, max_width=600):
    x, y = start_x, start_y
    space_width = 10  # Adjust space width as needed

    for number, alphabet in alphabet_gesture.items():
        text = f"{number}: {alphabet}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        if x + w > max_width:
            x = start_x
            y += line_height

        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x += w + space_width