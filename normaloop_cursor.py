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
    min_detection_confidence=0.1,  # Reduced from 0.7
    min_tracking_confidence=0.1    # Added tracking confidence
)
cap = cv2.VideoCapture(1)

gesture_time = time.time()
palm_gesture_time = time.time()

def get_hand_direction(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    # Calculate hand size/distance based on wrist to middle finger distance
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    hand_size = np.sqrt((wrist.x - middle_finger_tip.x)**2 + (wrist.y - middle_finger_tip.y)**2)
    
    # Lower base threshold and adjust scaling for better far detection
    base_threshold = 0.08
    distance_factor = max(0.3, min(2.0, hand_size * 6))  # More sensitive scaling
    adjusted_threshold = base_threshold / distance_factor
    
    movement = x_coords[4] - x_coords[20]  # Thumb vs pinky

    if movement > adjusted_threshold:
        return "LEFT"
    elif movement < -adjusted_threshold:
        return "RIGHT"
    return "NONE"

def detect_palm_gesture(landmarks):
    """Detect if hand is showing palm (all fingers extended)"""
    # Get finger tip landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Get finger base landmarks (for comparison)
    thumb_base = landmarks[3]
    index_base = landmarks[6]
    middle_base = landmarks[10]
    ring_base = landmarks[14]
    pinky_base = landmarks[18]
    
    # Check if all fingers are extended (tip is higher than base in y-coordinate)
    # Note: In OpenCV, y increases downward, so "higher" means smaller y value
    fingers_extended = []
    
    # Thumb (check x-coordinate since thumb moves horizontally)
    if thumb_tip.x > thumb_base.x:  # Right hand thumb extended
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
    
    # Other fingers (check y-coordinate)
    if index_tip.y < index_base.y:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
        
    if middle_tip.y < middle_base.y:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
        
    if ring_tip.y < ring_base.y:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
        
    if pinky_tip.y < pinky_base.y:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
    
    # All fingers must be extended for palm gesture
    return all(fingers_extended)

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
            is_palm = detect_palm_gesture(hand_landmarks.landmark)
            
            # Display distance info on screen
            cv2.putText(frame, f"Distance: {distance:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Direction: {direction}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Palm: {is_palm}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Handle swipe gestures
            if direction != "NONE" and (time.time() - gesture_time > 0.9):
                if direction == "LEFT":
                    pyautogui.hotkey('ctrl', 'right')
                    print(f"Swipe Left (Distance: {distance:.3f})")
                elif direction == "RIGHT":
                    pyautogui.hotkey('ctrl', 'left')
                    print(f"Swipe Right (Distance: {distance:.3f})")
                gesture_time = time.time()
            
            # Handle palm gesture for play/pause
            if is_palm and (time.time() - palm_gesture_time > 1.5):
                pyautogui.press('space')  # Space bar to play/pause
                print(f"Play/Pause (Distance: {distance:.3f})")
                palm_gesture_time = time.time()

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
