# Virtual Mouse Using Hand Gestures

This project implements a **Virtual Mouse** using **hand gestures** detected via a webcam. It uses **MediaPipe** for hand tracking and gesture recognition, and **PyAutoGUI** for controlling the mouse. The application allows users to interact with their computer using predefined hand gestures.

---

## Features

- **Mouse Control**:
  - Move the mouse cursor using your index finger.
  - Simulate mouse clicks by pinching your thumb and index finger together.

- **Gesture Recognition**:
  - Detects various hand gestures such as:
    - ✋ Open Palm: "Hello" or "Stop"
    - 🤟 "I Love You" gesture
    - 👍 Thumb Up: Increase volume
    - 👎 Thumb Down: Decrease volume
    - ✌️ Peace Sign: Switch between Gesture Mode and Mouse Mode
    - 👊 Fist: "No" or "Pain"
    - 🤞 Crossed Fingers: "Good Luck"
    - 👌 OK Sign: Confirmation
    - 🤙 Call Me Gesture
    - 🖐️ Palm Over Heart: "Thank You"

- **Swipe Detection**:
  - Swipe left or right to navigate slides or pages.

- **Audio Feedback**:
  - Provides audio feedback for certain gestures using text-to-speech.

---

## Requirements

- Python 3.7 or higher
- Webcam
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `pyautogui`
  - `pyttsx3`
  - `playsound`
  - `numpy`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Virtual-Mouse-using-Hand-Gesture.git
   cd Virtual-Mouse-using-Hand-Gesture
