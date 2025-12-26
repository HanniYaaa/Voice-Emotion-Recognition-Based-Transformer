# predict.py
import os
import pickle
import numpy as np
import librosa
import tensorflow as tf

SR = 16000
DURATION = 4.0

N_MELS = 64
HOP_LENGTH = 256
N_FFT = 512

MAX_LEN = int(np.ceil((SR * DURATION) / HOP_LENGTH))


# --------------------------------------------------
# Load Model + Label Encoder
# --------------------------------------------------
MODEL_PATH = "trained_transformer_best.keras"
LE_PATH = "label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)


# --------------------------------------------------
# Audio Process
# --------------------------------------------------
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


# --------------------------------------------------
# Predict function
# --------------------------------------------------
def predict_emotion(audio_path):
    y = load_audio(audio_path)
    feat = extract_log_mel(y)
    feat = np.expand_dims(feat, axis=0)   # (1, MAX_LEN, N_MELS)

    pred = model.predict(feat)[0]  # softmax vector
    idx = np.argmax(pred)
    emo = le.inverse_transform([idx])[0]

    return emo, pred


if __name__ == "__main__":
    test_file = r"dataset\studio001\middle\s001_middle_actor001_impro1_8.flac"
    emo, prob = predict_emotion(test_file)
    print("Emotion:", emo)
    print("Probabilities:", prob)
