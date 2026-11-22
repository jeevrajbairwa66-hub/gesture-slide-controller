import argparse
import csv
from collections import Counter
from pathlib import Path
from time import monotonic_ns
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "hand_landmarker.task"
DATASET_PATH = PROJECT_DIR / "gesture_dataset.csv"
WINDOW_NAME = "Gesture Dataset Collector"
MAX_FAILED_READS = 30

LABEL_KEYS = {
    ord("1"): "next",
    ord("2"): "back",
    ord("3"): "neutral",
}
FEATURE_NAMES = [
    f"{axis}{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]


def normalize_landmarks(landmarks):
    points = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32,
    )
    points -= points[0].copy()

    scale = np.linalg.norm(points, axis=1).max()
    if scale < 1e-6:
        return None

    return (points / scale).flatten()


def load_session_counts(session_id):
    if not DATASET_PATH.is_file():
        return Counter()

    with DATASET_PATH.open(encoding="utf-8", newline="") as file:
        return Counter(
            row["label"]
            for row in csv.DictReader(file)
            if row["session_id"] == session_id
        )


def collect_samples(camera, detector, session_id):
    counts = load_session_counts(session_id)
    selected_label = None
    status = "Select a gesture with 1, 2 or 3"
    failed_reads = 0
    last_timestamp_ms = -1

    with DATASET_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([*FEATURE_NAMES, "label", "session_id"])
            file.flush()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    print("Camera read failed 30 times in a row.", flush=True)
                    return 1
                if cv2.waitKey(50) & 0xFF == 27:
                    print("Collector closed: Esc pressed.", flush=True)
                    return 0
                continue

            failed_reads = 0
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = max(monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = detector.detect_for_video(image, timestamp_ms)

            features = None
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                features = normalize_landmarks(hand)

                for point in hand:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            lines = [
                "1: Next  2: Back  3: Neutral  S: Save  Esc: Quit",
                f"Session: {session_id} | Selected: {selected_label or 'None'}",
                f"Next: {counts['next']}  Back: {counts['back']}  Neutral: {counts['neutral']}",
                status,
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    frame, line, (10, 25 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Collector closed: Esc pressed.", flush=True)
                return 0

            if key in LABEL_KEYS:
                selected_label = LABEL_KEYS[key]
                status = "Show the selected pose and press S"
            elif key in (ord("s"), ord("S")):
                if selected_label is None:
                    status = "Select a label first"
                elif features is None:
                    status = "Show your hand before saving"
                else:
                    writer.writerow([*features.tolist(), selected_label, session_id])
                    file.flush()
                    counts[selected_label] += 1
                    status = f"Saved {selected_label}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="Name to reuse when resuming a recording")
    args = parser.parse_args()
    session_id = args.session or uuid4().hex

    if not MODEL_PATH.is_file():
        print("hand_landmarker.task ko collect_data.py ke paas rakho.")
        return 1

    print(f"Recording session: {session_id}", flush=True)

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        camera = cv2.VideoCapture(0)
        try:
            if not camera.isOpened():
                print("Camera open nahi hua.", flush=True)
                return 1
            return collect_samples(camera, detector, session_id)
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())