# Hand Gesture Control System

This project allows you to control media playback and volume on your computer using hand gestures captured by your webcam. It uses OpenCV and MediaPipe for hand tracking and PyAutoGUI for simulating keyboard presses.

## Features

The system supports various gestures for different controls. The main script (`gesturecontrol.py`) includes:

- **Play/Pause**: Open palm (5 fingers extended).
- **Mute/Unmute**: Fist (0 fingers extended).
- **Volume Control**: Move palm up/down vertically.
- **Next/Previous Track**: Swipe hand Left or Right.

### Variants

There are also variant scripts (`normaloop.py`, `normaloopP_downward.py`) tailored for specific controls (like Music Players) with different gesture mappings:
- **Play/Pause**: Open Palm.
- **Next Song**: Pointing Right (Index finger extended).
- **Previous Song**: Pointing Left or Down (depending on the script).

## Prerequisites

You need Python 3.x installed along with the following libraries:

- `opencv-python`
- `mediapipe`
- `pyautogui`
- `numpy`

## Installation

1. Clone this repository or download the folder.
2. Install the required dependencies:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

## Usage

1. Connect your webcam.
2. Run the main script:

```bash
python gesturecontrol.py
```

3. The camera window will open. Use your hand to control the media:
    - Keep your hand visible to the camera.
    - Perform the gestures described above.
    - Press `Esc` to exit the program.

## Troubleshooting

- Ensure you have good lighting.
- If gestures are not detected, try moving your hand closer or further from the camera.
- The scripts simulate keyboard shortcuts (e.g., `Space`, `m`, `Ctrl+Left`, `Shift+N`). Make sure the active window supports these shortcuts.

