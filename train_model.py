from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "gesture_dataset.csv"
MODEL_PATH = PROJECT_DIR / "gesture_model.joblib"

LABELS = ["next", "back", "neutral"]
FEATURE_COLUMNS = [
    f"{axis}{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]


def main():
    if not DATASET_PATH.is_file():
        print("gesture_dataset.csv nahi mili. Pehle samples record karo.")
        return

    data = pd.read_csv(DATASET_PATH, dtype={"session_id": str})
    required_columns = [*FEATURE_COLUMNS, "label", "session_id"]
    if not set(required_columns).issubset(data.columns):
        print("CSV mein hand points, label ya session_id columns missing hain.")
        return

    if data.empty:
        print("CSV khaali hai. Pehle gesture samples save karo.")
        return

    print("Samples per gesture:")
    print(data["label"].value_counts().reindex(LABELS, fill_value=0))

    # Use the most recently recorded session that contains all three gestures.
    complete_sessions = [
        session_id
        for session_id, samples in data.groupby("session_id", sort=False)
        if set(LABELS).issubset(samples["label"])
    ]
    if len(complete_sessions) < 2:
        print("Kam se kam 2 alag sessions mein teeno gestures record karo.")
        return

    test_session = complete_sessions[-1]
    test_data = data[data["session_id"] == test_session]
    train_data = data[data["session_id"] != test_session]

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1200,
        random_state=42,
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data["label"])

    predictions = model.predict(test_data[FEATURE_COLUMNS])
    accuracy = accuracy_score(test_data["label"], predictions)

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Test accuracy: {accuracy:.1%}")

    print("\nResults per gesture:")
    print(
        classification_report(
            test_data["label"], predictions, labels=LABELS, zero_division=0
        )
    )

    matrix = confusion_matrix(test_data["label"], predictions, labels=LABELS)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()