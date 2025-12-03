import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Much lower detection confidence for better detection at distance
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.05,  # Very low for far detection
    min_tracking_confidence=0.05,   # Very low for far detection
    model_complexity=1              # Use complex model for better accuracy
)
cap = cv2.VideoCapture(1)

palm_gesture_time = time.time()
next_gesture_time = time.time()
prev_gesture_time = time.time()

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
    
    # Calculate hand size for distance-based threshold adjustment
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    hand_size = np.sqrt((wrist.x - middle_finger_tip.x)**2 + (wrist.y - middle_finger_tip.y)**2)
    
    # Adjust finger detection threshold based on distance
    # Smaller hand = farther away = more lenient threshold
    distance_factor = max(0.1, min(1.0, hand_size * 2))
    finger_threshold = 0.02 / distance_factor  # Very small threshold for far detection
    
    # Check if all fingers are extended with distance-adjusted threshold
    fingers_extended = []
    
    # Thumb (check x-coordinate since thumb moves horizontally)
    if abs(thumb_tip.x - thumb_base.x) > finger_threshold:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
    
    # Other fingers (check y-coordinate with adjusted threshold)
    if (index_base.y - index_tip.y) > finger_threshold:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
        
    if (middle_base.y - middle_tip.y) > finger_threshold:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
        
    if (ring_base.y - ring_tip.y) > finger_threshold:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
        
    if (pinky_base.y - pinky_tip.y) > finger_threshold:
        fingers_extended.append(True)
    else:
        fingers_extended.append(False)
    
    # All fingers must be extended for palm gesture
    return all(fingers_extended)

def detect_next_gesture(landmarks):
    """Detect pointing right gesture (index finger extended, others closed)"""
    # Get finger tip landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Get finger base landmarks
    thumb_base = landmarks[3]
    index_base = landmarks[6]
    middle_base = landmarks[10]
    ring_base = landmarks[14]
    pinky_base = landmarks[18]
    
    # Calculate hand size for threshold adjustment
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    hand_size = np.sqrt((wrist.x - middle_finger_tip.x)**2 + (wrist.y - middle_finger_tip.y)**2)
    distance_factor = max(0.1, min(1.0, hand_size * 2))
    finger_threshold = 0.02 / distance_factor
    
    # Index finger should be extended
    index_extended = (index_base.y - index_tip.y) > finger_threshold
    
    # Other fingers should be closed
    middle_closed = (middle_base.y - middle_tip.y) <= finger_threshold
    ring_closed = (ring_base.y - ring_tip.y) <= finger_threshold
    pinky_closed = (pinky_base.y - pinky_tip.y) <= finger_threshold
    
    # Thumb can be either way
    thumb_ok = True
    
    return index_extended and middle_closed and ring_closed and pinky_closed and thumb_ok

def detect_prev_gesture(landmarks):
    """Detect pointing down gesture (index finger pointing down)"""
    # Get finger tip landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Get finger base landmarks
    thumb_base = landmarks[3]
    index_base = landmarks[6]
    middle_base = landmarks[10]
    ring_base = landmarks[14]
    pinky_base = landmarks[18]
    
    # Calculate hand size for threshold adjustment
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    hand_size = np.sqrt((wrist.x - middle_finger_tip.x)**2 + (wrist.y - middle_finger_tip.y)**2)
    distance_factor = max(0.1, min(1.0, hand_size * 2))
    finger_threshold = 0.02 / distance_factor
    
    # Index finger should be extended and pointing down
    # For pointing down: index tip y > index base y (tip is below base)
    index_pointing_down = (index_tip.y - index_base.y) > finger_threshold
    
    # Other fingers should be closed
    middle_closed = abs(middle_base.y - middle_tip.y) <= finger_threshold
    ring_closed = abs(ring_base.y - ring_tip.y) <= finger_threshold
    pinky_closed = abs(pinky_base.y - pinky_tip.y) <= finger_threshold
    
    # Thumb can be in any position
    thumb_ok = True
    
    # For previous: index pointing down + others closed
    return index_pointing_down and middle_closed and ring_closed and pinky_closed and thumb_ok

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
            distance = get_hand_distance(hand_landmarks.landmark)
            is_palm = detect_palm_gesture(hand_landmarks.landmark)
            is_next = detect_next_gesture(hand_landmarks.landmark)
            is_prev = detect_prev_gesture(hand_landmarks.landmark)
            
            # Display distance info on screen
            cv2.putText(frame, f"Distance: {distance:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Palm: {is_palm}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Next: {is_next}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Prev: {is_prev}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detailed debugging for previous gesture
            thumb_tip = hand_landmarks.landmark[4]
            thumb_base = hand_landmarks.landmark[3]
            index_tip = hand_landmarks.landmark[8]
            index_base = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            middle_base = hand_landmarks.landmark[10]
            
            thumb_diff = thumb_tip.x - thumb_base.x
            index_diff = index_base.y - index_tip.y
            middle_diff = middle_base.y - middle_tip.y
            
            cv2.putText(frame, f"Thumb: {thumb_diff:.3f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Index: {index_diff:.3f}", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Index Down: {index_tip.y - index_base.y:.3f}", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Handle palm gesture for play/pause
            if is_palm and (time.time() - palm_gesture_time > 1.5):
                pyautogui.press('space')  # Space bar to play/pause
                print(f"Play/Pause (Distance: {distance:.3f})")
                palm_gesture_time = time.time()
            
            # Handle next song gesture
            if is_next and (time.time() - next_gesture_time > 1.5):
                pyautogui.hotkey('shift', 'n')  # Shift+N for next song
                print(f"Next Song (Distance: {distance:.3f})")
                next_gesture_time = time.time()
            
            # Handle previous song gesture
            if is_prev and (time.time() - prev_gesture_time > 1.5):
                pyautogui.hotkey('shift', 'p')  # Shift+P for previous song
                print(f"Previous Song (Distance: {distance:.3f})")
                prev_gesture_time = time.time()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # Display "No hand detected" when no hand is found
        cv2.putText(frame, "No hand detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Palm=Play/Pause, Point=Next/Prev", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Gesture Control - Music Player", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
