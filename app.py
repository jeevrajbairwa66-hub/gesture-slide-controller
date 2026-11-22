import ctypes
from time import monotonic, monotonic_ns

import cv2
import joblib
import mediapipe as mp
import pandas as pd
import pyautogui

from collect_data import FEATURE_NAMES, MODEL_PATH, PROJECT_DIR, normalize_landmarks


ANN_PATH = PROJECT_DIR / "gesture_model.joblib"
WINDOW_NAME = "Gesture Slide Controller"
ARROW_KEYS = {"next": "right", "back": "left"}
MIN_CONFIDENCE = 0.85
STABLE_FRAMES = 5
RESET_FRAMES = 5
SWITCH_DELAY = 5

get_foreground_window = ctypes.windll.user32.GetForegroundWindow
get_foreground_window.restype = ctypes.c_void_p


def predict_gesture(model, hand):
    features = normalize_landmarks(hand)
    if features is None:
        return "uncertain", 0.0

    sample = pd.DataFrame([features], columns=FEATURE_NAMES)
    probabilities = model.predict_proba(sample)[0]
    best_index = int(probabilities.argmax())
    return model.classes_[best_index], probabilities[best_index]


def main():
    if not ANN_PATH.is_file():
        print("gesture_model.joblib nahi mili. Pehle train_model.py run karo.")
        return

    if not MODEL_PATH.is_file():
        print("hand_landmarker.task ko project folder mein rakho.")
        return

    model = joblib.load(ANN_PATH)
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    target_window = None
    preview_window = None
    switch_deadline = None
    armed = False
    reset_count = 0
    candidate = None
    candidate_count = 0
    failed_reads = 0
    last_timestamp_ms = -1

    with vision.HandLandmarker.create_from_options(options) as detector:
        camera = cv2.VideoCapture(0)

        try:
            if not camera.isOpened():
                print("Camera open nahi hua.")
                return

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    failed_reads += 1
                    if failed_reads >= 30:
                        print("Camera read failed 30 times in a row.")
                        break
                    if cv2.waitKey(50) & 0xFF == 27:
                        break
                    continue

                failed_reads = 0
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = max(monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(image, timestamp_ms)

                gesture = "no_hand"
                confidence = 0.0
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    gesture, confidence = predict_gesture(model, hand)

                    for point in hand:
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                foreground = get_foreground_window()
                now = monotonic()
                if switch_deadline is not None and now >= switch_deadline:
                    switch_deadline = None
                    if foreground and foreground != preview_window:
                        target_window = foreground

                if switch_deadline is not None:
                    remaining = max(1, int(switch_deadline - now + 0.99))
                    status = f"Switch to PowerPoint Slide Show: {remaining}s"
                elif target_window is None:
                    status = "Preview mode | Space: Enable"
                elif foreground != target_window:
                    status = "Paused | Return to the target window"
                elif armed:
                    status = "Ready | Show open palm or two fingers"
                else:
                    status = "Show fist or remove hand to enable next action"

                if target_window is not None and foreground == target_window:
                    is_reset = gesture == "no_hand" or (
                        gesture == "neutral" and confidence >= MIN_CONFIDENCE
                    )
                    reset_count = reset_count + 1 if is_reset else 0
                    if reset_count >= RESET_FRAMES:
                        armed = True

                    if armed and gesture in ARROW_KEYS and confidence >= MIN_CONFIDENCE:
                        if gesture == candidate:
                            candidate_count += 1
                        else:
                            candidate = gesture
                            candidate_count = 1

                        if candidate_count >= STABLE_FRAMES:
                            if get_foreground_window() == target_window:
                                pyautogui.press(ARROW_KEYS[gesture])
                                print(f"Slide command: {gesture}", flush=True)
                            armed = False
                            candidate = None
                            candidate_count = 0
                    else:
                        candidate = None
                        candidate_count = 0
                else:
                    armed = False
                    reset_count = 0
                    candidate = None
                    candidate_count = 0

                lines = [
                    f"Gesture: {gesture} | Confidence: {confidence:.0%}",
                    status,
                    "Space: Start/Stop slide control | Esc: Quit",
                ]
                for index, line in enumerate(lines):
                    cv2.putText(
                        frame, line, (12, 28 + index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                if key == ord(" "):
                    if target_window is not None or switch_deadline is not None:
                        target_window = None
                        switch_deadline = None
                    else:
                        preview_window = get_foreground_window()
                        switch_deadline = monotonic() + SWITCH_DELAY

        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()