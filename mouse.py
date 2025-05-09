import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import os
import threading
import pyttsx3
from playsound import playsound

# Make sure mouse control works
pyautogui.FAILSAFE = False

# Print screen resolution for debugging
print(f"Screen resolution: {pyautogui.size()}")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Create sounds directory if it doesn't exist
if not os.path.exists("sounds"):
    os.makedirs("sounds")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution
frame_width, frame_height = 640, 480
cap.set(3, frame_width)
cap.set(4, frame_height)
print(f"Camera resolution set to: {frame_width}x{frame_height}")

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Variables
mode = "gesture"  # Start in gesture mode
last_mode_switch = time.time()
mode_switch_cooldown = 1.0
click_threshold = 30
last_gesture_time = time.time()
gesture_cooldown = 1.5  # Time between recognizing gestures
last_recognized_gesture = None

# Smoothing variables
last_x, last_y = 0, 0
smoothing_factor = 0.2

# Function to speak text
def speak_text(text):
    threading.Thread(target=engine.say, args=(text,)).start()
    engine.runAndWait()

# Function to play sound if exists, otherwise speak text
def alert(sound_file, text):
    sound_path = os.path.join("sounds", sound_file)
    if os.path.exists(sound_path):
        threading.Thread(target=playsound, args=(sound_path,)).start()
    else:
        speak_text(text)

# Function to calculate distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to smooth the movement
def smooth_move(last_pos, current_pos, factor):
    return int(last_pos + (current_pos - last_pos) * factor)

# Function to check finger state (up or down)
def is_finger_up(hand_landmarks, finger_tip_idx, finger_pip_idx):
    return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_pip_idx].y

# Function to check thumb state (left/right depends on hand)
def is_thumb_up(hand_landmarks):
    return hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x

# Function to calculate finger angles
def finger_angle(hand_landmarks, p1, p2, p3):
    x1, y1 = hand_landmarks.landmark[p1].x, hand_landmarks.landmark[p1].y
    x2, y2 = hand_landmarks.landmark[p2].x, hand_landmarks.landmark[p2].y
    x3, y3 = hand_landmarks.landmark[p3].x, hand_landmarks.landmark[p3].y
    
    # Calculate vectors
    v1 = [x1 - x2, y1 - y2]
    v2 = [x3 - x2, y3 - y2]
    
    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Calculate magnitudes
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Calculate angle in radians and convert to degrees
    angle = math.acos(dot_product / (mag1 * mag2)) * 180 / math.pi
    
    return angle

# Gesture detector function
def detect_gesture(hand_landmarks):
    # Finger states (1 = up, 0 = down)
    finger_states = [0, 0, 0, 0, 0]  # thumb, index, middle, ring, pinky
    
    # Check thumb
    if is_thumb_up(hand_landmarks):
        finger_states[0] = 1
    
    # Check other fingers
    for i, (tip, pip) in enumerate(zip([8, 12, 16, 20], [6, 10, 14, 18])):
        if is_finger_up(hand_landmarks, tip, pip):
            finger_states[i+1] = 1
    
    # Get important landmark positions
    index_tip = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
    thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
    middle_tip = (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y)
    
    # Check distances
    thumb_index_dist = calculate_distance(
        hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y,
        hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
    )
    
    index_middle_dist = calculate_distance(
        hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y,
        hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
    )
    
    # Calculate angles
    thumb_angle = finger_angle(hand_landmarks, 4, 3, 2)
    index_angle = finger_angle(hand_landmarks, 8, 6, 5)
    middle_angle = finger_angle(hand_landmarks, 12, 10, 9)
    
    # Gesture detection logic
    
    # 1. "I Love You" sign (pinky, index up, thumb out)
    if finger_states == [1, 1, 0, 0, 1]:
        return "love_you"
    
    # 2. Open palm - all fingers up
    if sum(finger_states) == 5:
        return "open_palm"
    
    # 3. Crossed fingers - check index and middle finger crossing
    if finger_states[1] == 1 and finger_states[2] == 1 and index_middle_dist < 0.05:
        if abs(hand_landmarks.landmark[8].z - hand_landmarks.landmark[12].z) > 0.05:
            return "crossed_fingers"
    
    # 4. OK sign - thumb and index forming circle
    if thumb_index_dist < 0.05 and finger_states[2:] == [1, 1, 1]:
        return "ok_sign"
    
    # 5. Index finger up only
    if finger_states == [0, 1, 0, 0, 0]:
        return "index_up"
    
    # 6. Both hands up (can't detect with single hand model, but prepared for future)
    # Currently detecting as open palm
    
    # 7. Fist - all fingers down
    if sum(finger_states) == 0:
        return "fist"
    
    # 8. Palm facing out - same as open palm but check z-coordinate
    if sum(finger_states) == 5:
        # Check if palm is facing camera (negative z values)
        if hand_landmarks.landmark[9].z < -0.01:
            return "palm_out"
    
    # 9. Two fingers pointing together like walking
    if finger_states[1:3] == [1, 1] and finger_states[3:] == [0, 0]:
        if index_middle_dist < 0.04:
            return "walking_fingers"
    
    # 10. Flat palm toward mouth
    if sum(finger_states) >= 4:
        wrist_y = hand_landmarks.landmark[0].y
        middle_mcp_y = hand_landmarks.landmark[9].y
        if abs(wrist_y - middle_mcp_y) < 0.1:  # Horizontal palm
            # Check if close to mouth (upper part of the screen)
            if middle_mcp_y < 0.3:
                return "hungry"
    
    # 11. Thumb toward mouth
    if finger_states[0] == 1 and sum(finger_states[1:]) <= 1:
        if hand_landmarks.landmark[4].y < 0.3:  # Thumb near top of frame (mouth)
            return "thirsty"
    
    # 12. Point to ear
    if finger_states == [0, 1, 0, 0, 0]:  # Only index finger up
        if hand_landmarks.landmark[8].x < 0.2 or hand_landmarks.landmark[8].x > 0.8:
            return "listen_music"
    
    # 13. Palm over heart
    if sum(finger_states) >= 4:
        if 0.3 < hand_landmarks.landmark[9].x < 0.5 and 0.3 < hand_landmarks.landmark[9].y < 0.5:
            return "thank_you"
    
    # 14. Swipe detection is handled in the main loop by tracking movement
    
    # 15. Finger circle
    # This would need state tracking between frames
    
    # 16. Thumb up/down
    if finger_states == [1, 0, 0, 0, 0]:  # Only thumb up
        if hand_landmarks.landmark[4].y < hand_landmarks.landmark[9].y:
            return "thumb_up"
        else:
            return "thumb_down"
    
    # 17. Pinch open/close
    if thumb_index_dist < 0.05 and sum(finger_states[2:]) == 0:
        return "pinch"
    
    # 18. Finger gun (index + thumb)
    if finger_states == [1, 1, 0, 0, 0]:
        if thumb_angle > 30 and index_angle < 30:
            return "finger_gun"
    
    # 19. Two fingers up (peace sign)
    if finger_states == [0, 1, 1, 0, 0]:
        return "peace_sign"
    
    # 20. Call me hand (thumb and pinky)
    if finger_states == [1, 0, 0, 0, 1]:
        return "call_me"
    
    # 21. Two palms up together
    # Can't detect with single hand model
    
    return None

# Function to handle gesture actions
def handle_gesture(gesture, frame):
    global mode
    
    if gesture == "love_you":
        cv2.putText(frame, "I love you!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        speak_text("I love you")
        return True
    
    elif gesture == "open_palm":
        cv2.putText(frame, "Hello / Stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        speak_text("Hello")
        return True
    
    elif gesture == "crossed_fingers":
        cv2.putText(frame, "Good luck!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        speak_text("Good luck")
        return True
    
    elif gesture == "ok_sign":
        cv2.putText(frame, "OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        speak_text("OK")
        return True
    
    elif gesture == "index_up":
        cv2.putText(frame, "I need help", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        speak_text("I need help")
        return True
    
    elif gesture == "fist":
        cv2.putText(frame, "Pain / No", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        alert("alert.wav", "Pain signal detected")
        return True
    
    elif gesture == "palm_out":
        cv2.putText(frame, "Stop / Back off", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        speak_text("Stop")
        return True
    
    elif gesture == "walking_fingers":
        cv2.putText(frame, "Go to washroom", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        speak_text("Need to go to washroom")
        return True
    
    elif gesture == "hungry":
        cv2.putText(frame, "I'm hungry", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        speak_text("I'm hungry")
        return True
    
    elif gesture == "thirsty":
        cv2.putText(frame, "I'm thirsty", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        speak_text("I'm thirsty")
        return True
    
    elif gesture == "listen_music":
        cv2.putText(frame, "I want to listen to music", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
        speak_text("Play music")
        # You could add code here to actually start playing music
        return True
    
    elif gesture == "thank_you":
        cv2.putText(frame, "Thank you", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
        speak_text("Thank you")
        return True
    
    elif gesture == "thumb_up":
        cv2.putText(frame, "Volume Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        pyautogui.press('volumeup')
        return True
    
    elif gesture == "thumb_down":
        cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        pyautogui.press('volumedown')
        return True
    
    elif gesture == "pinch":
        cv2.putText(frame, "Zoom Control", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # Could implement zoom functionality here
        return True
    
    elif gesture == "finger_gun":
        cv2.putText(frame, "Select/Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if mode == "gesture":
            pyautogui.click()
        return True
    
    elif gesture == "peace_sign":
        cv2.putText(frame, "Mode Switch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        # Mode switching is handled elsewhere
        return True
    
    elif gesture == "call_me":
        cv2.putText(frame, "Call me", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        speak_text("Calling")
        # Could initiate a call here
        return True
    
    return False

# Track hand movement for swipe detection
last_positions = []
MAX_POSITIONS = 10

# Main loop
while True:
    # Read frame from camera
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        continue

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(rgb_frame)
    
    # Display mode
    if mode == "mouse":
        cv2.putText(frame, "MOUSE MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        cv2.putText(frame, "Peace sign to switch mode", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "GESTURE MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        cv2.putText(frame, "Peace sign to switch mode", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Check for hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect gesture
            gesture = detect_gesture(hand_landmarks)
            
            # Store hand position for swipe detection
            hand_center_x = int(hand_landmarks.landmark[9].x * frame_width)
            hand_center_y = int(hand_landmarks.landmark[9].y * frame_height)
            
            last_positions.append((hand_center_x, hand_center_y))
            if len(last_positions) > MAX_POSITIONS:
                last_positions.pop(0)
            
            # Detect swipe if we have enough positions
            if len(last_positions) == MAX_POSITIONS:
                start_x, start_y = last_positions[0]
                end_x, end_y = last_positions[-1]
                
                # Calculate horizontal and vertical movement
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                
                # Check if it's a swipe (significant horizontal movement)
                if abs(delta_x) > 100 and abs(delta_y) < 50:
                    current_time = time.time()
                    if current_time - last_gesture_time > gesture_cooldown:
                        if delta_x > 0:
                            cv2.putText(frame, "Swipe Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            pyautogui.hotkey('right')
                            last_gesture_time = current_time
                        else:
                            cv2.putText(frame, "Swipe Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            pyautogui.hotkey('left')
                            last_gesture_time = current_time
            
            # Count extended fingers
            fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
            count = 0
            
            # Check each finger
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:  # Thumb
                count += 1
            
            # Check other fingers
            for i in [8, 12, 16, 20]:  # Index, middle, ring, pinky tips
                if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i-2].y:
                    count += 1
            
            # Display finger count
            cv2.putText(frame, f"Fingers: {count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mode switching with peace sign
            current_time = time.time()
            if gesture == "peace_sign" and (current_time - last_mode_switch) > mode_switch_cooldown:
                mode = "mouse" if mode == "gesture" else "gesture"
                print(f"Switched to {mode.upper()} mode")
                speak_text(f"Switched to {mode} mode")
                last_mode_switch = current_time
            
            # Handle gestures if we're in gesture mode
            if mode == "gesture":
                if gesture and gesture != last_recognized_gesture:
                    current_time = time.time()
                    if current_time - last_gesture_time > gesture_cooldown:
                        if handle_gesture(gesture, frame):
                            last_gesture_time = current_time
                            last_recognized_gesture = gesture
            
            # Mouse control logic (when in mouse mode)
            if mode == "mouse":
                # Get index finger position
                index_x = int(hand_landmarks.landmark[8].x * frame_width)
                index_y = int(hand_landmarks.landmark[8].y * frame_height)
                
                # Mark index finger position
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), -1)
                
                # Direct mapping from camera to screen (with scaling)
                mouse_x = int(hand_landmarks.landmark[8].x * screen_width)
                mouse_y = int(hand_landmarks.landmark[8].y * screen_height)
                
                # Smooth mouse movement
                mouse_x = smooth_move(last_x, mouse_x, smoothing_factor)
                mouse_y = smooth_move(last_y, mouse_y, smoothing_factor)
                
                # Ensure coordinates are within screen bounds
                mouse_x = max(0, min(mouse_x, screen_width - 1))
                mouse_y = max(0, min(mouse_y, screen_height - 1))
                
                # Move the cursor - using absolute position
                try:
                    # Use direct OS level control 
                    pyautogui.moveTo(mouse_x, mouse_y, _pause=False)
                    
                    # Update last position for smoothing
                    last_x, last_y = mouse_x, mouse_y
                except Exception as e:
                    print(f"Error moving mouse: {e}")
                
                # Show mouse coordinates on screen
                cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Click detection
                thumb_x = int(hand_landmarks.landmark[4].x * frame_width)
                thumb_y = int(hand_landmarks.landmark[4].y * frame_height)
                
                # Calculate distance between thumb and index finger
                distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
                
                # Show distance
                cv2.putText(frame, f"Distance: {int(distance)}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Click if distance is below threshold
                if distance < click_threshold:
                    cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), -1)
                    print("Clicking")
                    pyautogui.click()
                    time.sleep(0.5)  # Prevent multiple clicks
                
                # Also check for special gestures even in mouse mode
                if gesture == "finger_gun":
                    pyautogui.click()
                    time.sleep(0.5)
    
    # Show current gesture if detected
    if last_recognized_gesture:
        cv2.putText(frame, f"Gesture: {last_recognized_gesture}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Enhanced Gesture Control", frame)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("Closing application")
cap.release()
cv2.destroyAllWindows()