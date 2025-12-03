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
y_positions = []

def get_hand_direction(landmarks):
    x_coords = [lm.x for lm in landmarks]
    movement = x_coords[4] - x_coords[20]  # Thumb vs pinky

    if movement > 0.2:
        return "LEFT"
    elif movement < -0.2:
        return "RIGHT"
    return "NONE"

def count_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]
    count = 0
    for tip in tips[1:]:  # Skip thumb
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    if landmarks[4].x < landmarks[3].x:
        count += 1
    return count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            direction = get_hand_direction(lm)
            fingers = count_fingers(lm)

            # Track Y position of wrist/palm base
            y_positions.append(lm[0].y)
            if len(y_positions) > 5:
                y_positions.pop(0)

            # Swipe Left/Right
            if direction != "NONE" and direction != prev_gesture and (time.time() - gesture_time > 1):
                if direction == "LEFT":
                    pyautogui.hotkey('ctrl', 'left')
                    print("Swipe Left")
                elif direction == "RIGHT":
                    pyautogui.hotkey('ctrl', 'right')
                    print("Swipe Right")
                prev_gesture = direction
                gesture_time = time.time()

            # Play/Pause (5 fingers)
            if fingers == 5 and (time.time() - gesture_time > 1):
                pyautogui.press("space")
                print("Play/Pause")
                gesture_time = time.time()

            # Mute/Unmute (Fist)
            if fingers == 0 and (time.time() - gesture_time > 1):
                pyautogui.press("m")
                print("Mute/Unmute")
                gesture_time = time.time()

            # Volume Up/Down (based on palm movement)
            if len(y_positions) == 5:
                diff = y_positions[-1] - y_positions[0]
                if diff > 0.05:
                    pyautogui.press("volumedown")
                    print("Volume Down")
                    y_positions.clear()
                elif diff < -0.05:
                    pyautogui.press("volumeup")
                    print("Volume Up")
                    y_positions.clear()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Swipe Control", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
