"""Dense autoencoder that compresses customer features into an 8-D latent space."""
from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
LATENT_DIM = 8
MAX_EPOCHS = 150


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source so training is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # older TF builds
        pass


def build_autoencoder(n_features: int) -> tuple[keras.Model, keras.Model]:
    """Return (autoencoder, encoder) sharing the same encoder layers."""
    inputs = keras.Input(shape=(n_features,), name="features")
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x)
    latent = layers.Dense(LATENT_DIM, activation="relu", name="latent")(x)

    y = layers.Dense(16, activation="relu")(latent)
    y = layers.Dense(32, activation="relu")(y)
    outputs = layers.Dense(n_features, activation="linear", name="reconstruction")(y)

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return autoencoder, encoder


def train_autoencoder(X: np.ndarray, encoder_path: str | Path) -> dict:
    """Train the autoencoder, save the encoder and return latent codes, losses and recon errors."""
    set_seeds()
    autoencoder, encoder = build_autoencoder(X.shape[1])
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    # One progress line every 10 epochs (read by server.py to drive the upload progress bar).
    progress = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: (epoch + 1) % 10 == 0 and print(
            f"[autoencoder] epoch {epoch + 1}/{MAX_EPOCHS} loss={logs['loss']:.4f} "
            f"val_loss={logs['val_loss']:.4f}", flush=True)
    )
    history = autoencoder.fit(
        X, X, epochs=MAX_EPOCHS, batch_size=32, validation_split=0.2,
        shuffle=True, callbacks=[early, progress], verbose=0,
    )
    encoder.save(encoder_path)

    latent = encoder.predict(X, verbose=0)
    recon = autoencoder.predict(X, verbose=0)
    recon_error = np.mean(np.square(X - recon), axis=1)

    loss = [float(v) for v in history.history["loss"]]
    val_loss = [float(v) for v in history.history["val_loss"]]
    best_epoch = int(np.argmin(val_loss)) + 1
    print(f"[autoencoder] trained {len(loss)} epochs (best epoch {best_epoch}, "
          f"val_loss {min(val_loss):.4f}); encoder saved to {Path(encoder_path).name}")
    return {
        "latent": latent,
        "recon_error": recon_error,
        "history": {"loss": loss, "val_loss": val_loss, "best_epoch": best_epoch},
    }
