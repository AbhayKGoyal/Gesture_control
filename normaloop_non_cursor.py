import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lower detection confidence for better detection at distance
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.5,  # Reduced from 0.7
    min_tracking_confidence=0.5    # Added tracking confidence
)
cap = cv2.VideoCapture(1)

gesture_time = time.time()

def get_hand_direction(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    # Calculate hand size/distance based on wrist to middle finger distance
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    hand_size = np.sqrt((wrist.x - middle_finger_tip.x)**2 + (wrist.y - middle_finger_tip.y)**2)
    
    # Adjust threshold based on hand size (distance from camera)
    # Smaller hand size = farther from camera, so use smaller threshold
    base_threshold = 0.15
    distance_factor = max(0.5, min(2.0, hand_size * 3))  # Scale factor based on distance
    adjusted_threshold = base_threshold / distance_factor
    
    movement = x_coords[4] - x_coords[20]  # Thumb vs pinky

    if movement > adjusted_threshold:
        return "LEFT"
    elif movement < -adjusted_threshold:
        return "RIGHT"
    return "NONE"

def get_hand_distance(landmarks):
    """Calculate approximate distance of hand from camera"""
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    hand_size = np.sqrt((wrist.x - middle_finger_tip.x)**2 + (wrist.y - middle_finger_tip.y)**2)
    return hand_size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            direction = get_hand_direction(hand_landmarks.landmark)
            distance = get_hand_distance(hand_landmarks.landmark)
            
            # Display distance info on screen
            cv2.putText(frame, f"Distance: {distance:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Direction: {direction}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Allow same gesture every 0.7s
            if direction != "NONE" and (time.time() - gesture_time > 0.9):
                if direction == "LEFT":
                    pyautogui.hotkey('ctrl', 'right')
                    print(f"Swipe Left (Distance: {distance:.3f})")
                elif direction == "RIGHT":
                    pyautogui.hotkey('ctrl', 'left')
                    print(f"Swipe Right (Distance: {distance:.3f})")
                gesture_time = time.time()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # Display "No hand detected" when no hand is found
        cv2.putText(frame, "No hand detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Gesture Swipe Control", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
