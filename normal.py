import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Track hand  
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(1)

prev_gesture = None
gesture_time = time.time()

def get_hand_direction(landmarks):
    x_coords = [lm.x for lm in landmarks]
    movement = x_coords[4] - x_coords[20]  # Thumb vs pinky

    if movement > 0.2:
        return "RIGHT"
    elif movement < -0.2:
        return "LEFT"
    return "NONE"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            direction = get_hand_direction(hand_landmarks.landmark)

            # Debounce to avoid multiple triggers
            if direction != "NONE" and direction != prev_gesture and (time.time() - gesture_time > 1):
                if direction == "LEFT":
                    pyautogui.hotkey('ctrl', 'left')
                    print("Swipe Left")
                elif direction == "RIGHT":
                    pyautogui.hotkey('ctrl', 'right')
                    print("Swipe Right")
                prev_gesture = direction
                gesture_time = time.time()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Swipe Control", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
