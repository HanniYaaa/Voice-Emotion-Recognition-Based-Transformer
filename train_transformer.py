# train_transformer_ready_improved.py
import os
import json
import math
import random
import pickle
from glob import glob

import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Try AdamW
try:
    import tensorflow_addons as tfa
    USE_ADAMW = True
except:
    USE_ADAMW = False


# ==========================================================
# Config
# ==========================================================
DATASET_DIR = "dataset"
LABEL_JSON = "emotion_label.json"

SR = 16000
DURATION = 4.0

N_MELS = 64
HOP_LENGTH = 256
N_FFT = 512

MAX_LEN = int(math.ceil((SR * DURATION) / HOP_LENGTH))

BATCH_SIZE = 32
EPOCHS = 100
RANDOM_SEED = 42
NUM_CLASSES = 5

EMO_MAPPING = {
    "Neutral": "Neutral",
    "Angry": "Angry",
    "Happy": "Happy",
    "Sad": "Sad",
    "Frustrated": "Frustrated"
}


# ==========================================================
# Positional Encoding
# ==========================================================
def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    idx = np.arange(d_model)[None, :]

    angles = pos / np.power(10000, (2 * (idx // 2)) / d_model)

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.convert_to_tensor(angles, dtype=tf.float32)


# ==========================================================
# Audio processing
# ==========================================================
def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)

    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
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

    # Padding / Trim
    if log_mel.shape[0] > MAX_LEN:
        log_mel = log_mel[:MAX_LEN]
    elif log_mel.shape[0] < MAX_LEN:
        pad = MAX_LEN - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="constant")

    return log_mel.astype("float32")


# ==========================================================
# Transformer Encoder
# ==========================================================
def transformer_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    # ---- Multi-head Attention ----
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)

    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # ---- Feed Forward Network ----
    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)

    out = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return out


def build_transformer_model(
    input_shape=(MAX_LEN, N_MELS),
    d_model=256,
    num_heads=8,
    ff_dim=512,
    num_layers=6,
    dropout=0.3,
    num_classes=NUM_CLASSES
):

    inp = layers.Input(shape=input_shape)

    # Projection to d_model
    x = layers.Dense(d_model)(inp)

    # Add Positional Encoding
    pos = get_positional_encoding(input_shape[0], d_model)
    x = x + pos

    # Transformer stacks
    for _ in range(num_layers):
        x = transformer_block(x, d_model, num_heads, ff_dim, dropout)

    # Pooling
    avg = layers.GlobalAveragePooling1D()(x)
    mx = layers.GlobalMaxPooling1D()(x)
    x = layers.concatenate([avg, mx])

    # Classifier
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out)


# ==========================================================
# Label JSON
# ==========================================================
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


def gather_files_and_labels(dataset_root, label_map):
    X, y = [], []

    for p in glob(os.path.join(dataset_root, "**", "*.flac"), recursive=True):
        fname = os.path.basename(p)

        if fname in label_map:
            emo = label_map[fname]

            if emo in EMO_MAPPING:
                X.append(p)
                y.append(EMO_MAPPING[emo])
            else:
                # fuzzy match
                for k in EMO_MAPPING:
                    if k.lower() in emo.lower():
                        X.append(p)
                        y.append(k)
                        break

    return X, y


# ==========================================================
# Dataset TF
# ==========================================================
def preprocess(path, label):

    def _load_fn(p):
        p = tf.compat.as_str(p.numpy())
        y = load_audio(p)
        feat = extract_log_mel(y)
        return feat

    feat = tf.py_function(_load_fn, [path], tf.float32)
    feat.set_shape((MAX_LEN, N_MELS))

    return feat, label


def make_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=RANDOM_SEED)

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ==========================================================
# MAIN
# ==========================================================
def main():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("Loading label JSON")
    label_map = parse_label_json(LABEL_JSON)

    print("Gathering audio files...")
    X_paths, y_labels_str = gather_files_and_labels(DATASET_DIR, label_map)
    print("Found:", len(X_paths))

    if len(X_paths) == 0:
        raise RuntimeError("No labeled data found!")

    # Label Encoding
    le = LabelEncoder()
    y_int = le.fit_transform(y_labels_str)

    train_p, val_p, train_y, val_y = train_test_split(
        X_paths,
        y_int,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_int
    )

    train_ds = make_dataset(train_p, train_y)
    val_ds = make_dataset(val_p, val_y, shuffle=False)

    # Build model
    model = build_transformer_model()
    model.summary()

    # Optimizer
    if USE_ADAMW:
        opt = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    else:
        opt = optimizers.Adam(1e-4)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        "trained_transformer_best.keras",   # เซฟเป็น .keras ได้ปกติ
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    csv = callbacks.CSVLogger("training_log.csv")

    cbs = [checkpoint, reduce_lr, early, csv]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cbs
    )

    # Save artifacts
    # --- save เป็น Keras format (.keras) หรือ SavedModel directory ---
    model.save("trained_transformer_final.keras")   # <-- เปลี่ยนจาก .h5 เป็น .keras
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("Training completed!")


if __name__ == "__main__":
    main()
