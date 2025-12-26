import os
import json
import pickle
import numpy as np
import librosa
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import tensorflow as tf

SR = 16000
DURATION = 4.0

N_MELS = 64
HOP_LENGTH = 256
N_FFT = 512
MAX_LEN = int(np.ceil((SR * DURATION) / HOP_LENGTH))

DATASET_DIR = "dataset"
LABEL_JSON = "emotion_label.json"

MODEL_PATH = "trained_transformer_best.keras"
LE_PATH = "label_encoder.pkl"


# ----------------------------------------------------------
# Load Model + Label Encoder
# ----------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LE_PATH, "rb") as f:
    le = pickle.load(f)


# ----------------------------------------------------------
# Audio functions
# ----------------------------------------------------------
def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)

    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    return y


def extract_log_mel(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T

    if log_mel.shape[0] > MAX_LEN:
        log_mel = log_mel[:MAX_LEN]
    else:
        log_mel = np.pad(log_mel, ((0, MAX_LEN - log_mel.shape[0]), (0, 0)))

    return log_mel.astype("float32")


# ----------------------------------------------------------
# Parse label JSON
# ----------------------------------------------------------
def parse_label_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mapping = {}
    for k, arr in meta.items():
        if not arr:
            continue

        item = arr[0]
        emo = item.get("assigned_emo") or item.get("majority_emo")

        if emo is None:
            if "annotated" in item and item["annotated"]:
                emo = item["annotated"][0][0]

        if emo:
            mapping[k] = emo

    return mapping


# ----------------------------------------------------------
# Collect validation files
# ----------------------------------------------------------
def gather_files_and_labels(dataset_root, label_map):
    X, y = [], []

    for p in glob(os.path.join(dataset_root, "**", "*.flac"), recursive=True):
        fname = os.path.basename(p)
        if fname in label_map:
            X.append(p)
            y.append(label_map[fname])

    return X, y


# ----------------------------------------------------------
# PLOT CONFUSION MATRIX
# ----------------------------------------------------------
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.show()


# ----------------------------------------------------------
# PLOT ACCURACY BAR
# ----------------------------------------------------------
def plot_accuracy(acc):
    plt.figure(figsize=(4, 6))
    plt.bar(["Accuracy"], [acc])
    plt.ylim(0, 1)
    plt.title(f"Accuracy = {acc:.4f}")
    plt.tight_layout()
    plt.savefig("accuracy_bar.png", dpi=200)
    plt.show()


# ----------------------------------------------------------
# EVALUATE
# ----------------------------------------------------------
def evaluate():
    label_map = parse_label_json(LABEL_JSON)

    X_paths, y_str = gather_files_and_labels(DATASET_DIR, label_map)

    y_true_int = le.transform(y_str)
    y_pred_int = []

    print(f"Evaluating {len(X_paths)} files...")

    for path in X_paths:
        y = load_audio(path)
        feat = extract_log_mel(y)
        feat = np.expand_dims(feat, axis=0)

        pred = model.predict(feat, verbose=0)[0]
        idx = np.argmax(pred)
        y_pred_int.append(idx)

    acc = accuracy_score(y_true_int, y_pred_int)
    print("Accuracy =", acc)

    # Plot accuracy
    plot_accuracy(acc)

    # Classification Report (text)
    print("\nClassification Report:")
    print(classification_report(le.inverse_transform(y_true_int),
                               le.inverse_transform(y_pred_int)))

    # Confusion Matrix
    cm = confusion_matrix(
        le.inverse_transform(y_true_int),
        le.inverse_transform(y_pred_int)
    )
    plot_confusion_matrix(cm, le.classes_)

    print("\nSaved images:")
    print(" - confusion_matrix.png")
    print(" - accuracy_bar.png")


if __name__ == "__main__":
    evaluate()
