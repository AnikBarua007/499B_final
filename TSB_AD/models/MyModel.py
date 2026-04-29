import random

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


def _require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as exc:
        raise ImportError(
            "MyModel requires tensorflow. Install it separately to use the "
            "notebook-derived hybrid detector."
        ) from exc
    return tf, keras, layers


def _to_2d_array(data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError("MyModel expects input with shape (n_samples, n_features).")
    return data


class MyModel(BaseDetector):
    def __init__(
        self,
        window_size=100,
        stride=5,
        batch_size=128,         # doubled for throughput
        ae_epochs=15,           # halved — early stopping catches convergence
        lr=1e-3,               # higher LR to converge in fewer epochs
        dropout=0.1,
        mask_ratio=0.25,
        validation_size=0.2,
        recon_weight=1.0,
        noise_std=0.005,
        scale_min=0.95,
        scale_max=1.05,
        seed=42,
        verbose=0,
        finetune_epochs=1,      # 1 epoch is enough for fine-tuning
        score_sharpness=2.0,
    ):
        self.model_name = "MyModel"
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.ae_epochs = int(ae_epochs)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.mask_ratio = float(mask_ratio)
        self.validation_size = float(validation_size)
        self.recon_weight = float(recon_weight)
        self.noise_std = float(noise_std)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.seed = int(seed)
        self.verbose = int(verbose)
        self.finetune_epochs = int(finetune_epochs)
        self.score_sharpness = float(score_sharpness)
        self._validate_params()

        self.scaler = None
        self.autoencoder = None
        self.encoder = None
        self.prototype_ = None
        self.recon_min_ = 0.0
        self.recon_max_ = 1.0
        self.tf = None
        self.keras = None
        self.layers = None

    def _validate_params(self):
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0.")
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.ae_epochs <= 0:
            raise ValueError("ae_epochs must be > 0.")
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (0.0 <= self.mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0, 1].")
        if not (0.0 <= self.validation_size < 1.0):
            raise ValueError("validation_size must be in [0, 1).")
        if self.recon_weight < 0:
            raise ValueError("recon_weight must be >= 0.")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0.")
        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("scale_min and scale_max must be > 0.")
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must be <= scale_max.")
        if self.finetune_epochs < 0:
            raise ValueError("finetune_epochs must be >= 0.")
        if self.score_sharpness < 0:
            raise ValueError("score_sharpness must be >= 0.")

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.tf.random.set_seed(self.seed)

    def _ensure_min_length(self, data):
        if len(data) >= self.window_size:
            return data
        pad_len = self.window_size - len(data)
        pad = np.repeat(data[:1], pad_len, axis=0)
        return np.concatenate([pad, data], axis=0)

    def _make_windows(self, data, stride=None):
        if stride is None:
            stride = self.stride
        data = self._ensure_min_length(data)
        starts = list(range(0, len(data) - self.window_size + 1, stride))
        if not starts or starts[-1] != len(data) - self.window_size:
            starts.append(len(data) - self.window_size)
        windows = np.stack([data[s : s + self.window_size] for s in starts]).astype(np.float32)
        return windows, np.asarray(starts, dtype=np.int32), len(data)

    def _scale_fit(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        return self.scaler.transform(data).astype(np.float32)

    def _scale_transform(self, data):
        return self.scaler.transform(data).astype(np.float32)

    # [Improvement 7] Learning rate warmup + cosine decay
    def _get_lr(self, epoch):
        warmup = 3
        if epoch < warmup:
            return self.lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, self.ae_epochs - warmup)
        return self.lr * 0.5 * (1.0 + np.cos(np.pi * progress))

    def _build_autoencoder(self, timesteps, num_features):
        tf, keras, layers = self.tf, self.keras, self.layers

        inputs = keras.Input(shape=(timesteps, num_features))

        # Lighter encoder — 128 channels instead of 256 (4x fewer FLOPs in conv)
        x = layers.Conv1D(32, 5, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
        x = layers.Conv1D(64, 5, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Dropout(self.dropout)(x)

        res = x
        x = layers.Conv1D(128, 3, padding="same", activation="relu", strides=2, kernel_initializer="he_normal")(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        res_down = layers.Conv1D(128, 1, strides=2, padding="same")(res)
        x = layers.Add()([x, res_down])
        x = layers.Dropout(self.dropout)(x)

        res = x
        x = layers.Conv1D(128, 3, padding="same", activation="relu", strides=2, kernel_initializer="he_normal")(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        res_down = layers.Conv1D(128, 1, strides=2, padding="same")(res)
        x = layers.Add()([x, res_down])

        # [Improvement 2] GRU at bottleneck — smaller for speed
        x = layers.GRU(64, return_sequences=True)(x)

        bottleneck = x

        # Lighter decoder
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Conv1D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        outputs = layers.Conv1D(num_features, 3, padding="same")(x)
        outputs = layers.Lambda(lambda t: t[:, :timesteps, :])(outputs)

        autoencoder = keras.Model(inputs, outputs)
        encoder = keras.Model(inputs, bottleneck)
        optimizer = keras.optimizers.Adam(self.lr)
        return autoencoder, encoder, optimizer

    def _augment_windows(self, x):
        tf = self.tf
        x_noisy = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std)
        scales = tf.random.uniform(
            [tf.shape(x)[0], 1, 1], self.scale_min, self.scale_max
        )
        return x_noisy * scales

    def _apply_random_mask(self, x):
        tf = self.tf
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        mask = tf.cast(tf.random.uniform((b, t), 0, 1) < self.mask_ratio, tf.float32)
        x_masked = x * tf.expand_dims(1 - mask, -1)
        return x_masked, tf.expand_dims(mask, -1)

    def _masked_loss(self, x, recon, mask):
        tf = self.tf
        diff = (x - recon) * mask
        feature_count = tf.cast(tf.shape(x)[-1], tf.float32)
        denom = (tf.reduce_sum(mask) * feature_count) + 1e-8
        return tf.reduce_sum(tf.square(diff)) / denom

    def _fit_autoencoder(self, train_windows, val_windows):
        tf = self.tf
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_windows)
            .shuffle(max(len(train_windows), 1), seed=self.seed, reshuffle_each_iteration=True)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices(val_windows)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        @tf.function
        def train_step(batch_x):
            x_aug = self._augment_windows(batch_x)
            x_masked, mask = self._apply_random_mask(x_aug)
            with tf.GradientTape() as tape:
                recon = self.autoencoder(x_masked, training=True)
                loss = self._masked_loss(batch_x, recon, mask)
            grads = tape.gradient(loss, self.autoencoder.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
            return loss

        # Clean validation: no augmentation or masking noise
        @tf.function
        def val_step(batch_x):
            recon = self.autoencoder(batch_x, training=False)
            return tf.reduce_mean(tf.square(batch_x - recon))

        best_weights = None
        best_val = np.inf
        patience = 5           # [Improvement 6] increased from 3
        min_delta = 1e-5       # [Improvement 6] minimum improvement threshold
        wait = 0

        # --- Phase 1: Masked autoencoder pretraining ---
        for epoch in range(self.ae_epochs):
            # [Improvement 7] Apply LR schedule
            new_lr = self._get_lr(epoch)
            self.optimizer.learning_rate.assign(new_lr)

            train_losses = [float(train_step(batch).numpy()) for batch in train_ds]
            val_losses = [float(val_step(batch).numpy()) for batch in val_ds]
            val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))

            if self.verbose:
                train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                print(
                    f"MyModel epoch {epoch + 1:02d} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={new_lr:.2e}"
                )

            if val_loss < best_val - min_delta:
                best_val = val_loss
                wait = 0
                best_weights = self.autoencoder.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_weights is not None:
            self.autoencoder.set_weights(best_weights)

        # --- [Improvement 4] Phase 2: Full-reconstruction fine-tuning ---
        if self.finetune_epochs > 0:
            ft_lr = self.lr * 0.1  # lower LR for fine-tuning
            self.optimizer.learning_rate.assign(ft_lr)

            @tf.function
            def finetune_step(batch_x):
                with tf.GradientTape() as tape:
                    recon = self.autoencoder(batch_x, training=True)
                    loss = tf.reduce_mean(tf.square(batch_x - recon))
                grads = tape.gradient(loss, self.autoencoder.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
                return loss

            for epoch in range(self.finetune_epochs):
                ft_losses = [float(finetune_step(batch).numpy()) for batch in train_ds]
                if self.verbose:
                    print(
                        f"MyModel finetune {epoch + 1:02d} | "
                        f"recon_loss={float(np.mean(ft_losses)):.6f}"
                    )

    def _get_embeddings(self, windows):
        tf = self.tf
        n = len(windows)
        bs = self.batch_size * 4  # larger batch for inference (no gradients)
        parts = []
        for i in range(0, n, bs):
            chunk = windows[i : i + bs]
            emb = self.encoder(chunk, training=False)
            emb = tf.reduce_mean(emb, axis=1)
            emb = tf.math.l2_normalize(emb, axis=1)
            parts.append(emb.numpy())
        return np.concatenate(parts, axis=0)

    def _window_scores(self, windows):
        tf = self.tf
        n = len(windows)
        bs = self.batch_size * 4  # larger batch — no gradients needed

        # Single pass: get both embeddings and reconstruction in one loop
        all_emb = []
        recon_scores = np.empty(n, dtype=np.float32)
        for i in range(0, n, bs):
            chunk = windows[i : i + bs]
            # Direct model call — much faster than .predict() per-batch
            emb = self.encoder(chunk, training=False)
            emb = tf.reduce_mean(emb, axis=1)
            emb = tf.math.l2_normalize(emb, axis=1)
            all_emb.append(emb.numpy())

            recon = self.autoencoder(chunk, training=False)
            chunk_np = chunk.numpy() if hasattr(chunk, "numpy") else chunk
            recon_np = recon.numpy() if hasattr(recon, "numpy") else recon
            recon_scores[i : i + bs] = np.mean((chunk_np - recon_np) ** 2, axis=(1, 2))

        embeddings = np.concatenate(all_emb, axis=0)
        proto_scores = np.linalg.norm(embeddings - self.prototype_, axis=1)

        recon_scores = (recon_scores - self.recon_min_) / (self.recon_max_ - self.recon_min_ + 1e-8)
        recon_scores = np.clip(recon_scores, 0.0, 1.0)

        return proto_scores + self.recon_weight * recon_scores

    def _window_scores_detailed(self, windows):
        """Return (combined, recon_scores, proto_scores) per window."""
        tf = self.tf
        n = len(windows)
        bs = self.batch_size * 4

        all_emb = []
        recon_scores = np.empty(n, dtype=np.float32)
        for i in range(0, n, bs):
            chunk = windows[i : i + bs]
            emb = self.encoder(chunk, training=False)
            emb = tf.reduce_mean(emb, axis=1)
            emb = tf.math.l2_normalize(emb, axis=1)
            all_emb.append(emb.numpy())

            recon = self.autoencoder(chunk, training=False)
            chunk_np = chunk.numpy() if hasattr(chunk, "numpy") else chunk
            recon_np = recon.numpy() if hasattr(recon, "numpy") else recon
            recon_scores[i : i + bs] = np.mean((chunk_np - recon_np) ** 2, axis=(1, 2))

        embeddings = np.concatenate(all_emb, axis=0)
        proto_scores = np.linalg.norm(embeddings - self.prototype_, axis=1)

        recon_scores = (recon_scores - self.recon_min_) / (self.recon_max_ - self.recon_min_ + 1e-8)
        recon_scores = np.clip(recon_scores, 0.0, 1.0)

        combined = proto_scores + self.recon_weight * recon_scores
        return combined, recon_scores, proto_scores

    # [Improvement 1] Max-aggregation instead of mean — preserves anomaly peaks
    def _pointwise_scores(self, window_scores, starts, padded_len, original_len):
        point_scores = np.full(padded_len, -np.inf, dtype=np.float32)
        for score, start in zip(window_scores, starts):
            end = start + self.window_size
            point_scores[start:end] = np.maximum(point_scores[start:end], score)
        point_scores[point_scores == -np.inf] = 0.0
        return point_scores[-original_len:]

    # [Improvement 5] Score sharpening — amplify anomaly/normal separation
    def _sharpen_scores(self, scores):
        if self.score_sharpness <= 0:
            return scores
        # Normalize to [0, 1] first to keep exponent stable
        s_min = scores.min()
        s_max = scores.max()
        if s_max - s_min < 1e-12:
            return scores
        normed = (scores - s_min) / (s_max - s_min)
        sharpened = np.exp(normed * self.score_sharpness) - 1.0
        return sharpened

    def fit(self, X, y=None):
        self.tf, self.keras, self.layers = _require_tensorflow()
        self._set_seed()

        X = _to_2d_array(X)
        X_scaled = self._scale_fit(X)
        train_windows, _, _ = self._make_windows(X_scaled)

        split_idx = int((1 - self.validation_size) * len(train_windows))
        if split_idx <= 0 or split_idx >= len(train_windows):
            train_split = train_windows
            val_split = train_windows
        else:
            train_split = train_windows[:split_idx]
            val_split = train_windows[split_idx:]
            if len(val_split) == 0:
                val_split = train_split

        self.autoencoder, self.encoder, self.optimizer = self._build_autoencoder(
            timesteps=self.window_size,
            num_features=X.shape[1],
        )
        self._fit_autoencoder(train_split, val_split)

        train_embeddings = self._get_embeddings(train_windows)
        self.prototype_ = np.mean(train_embeddings, axis=0)

        recon = self.autoencoder.predict(train_windows, verbose=0)
        train_recon_scores = np.mean((train_windows - recon) ** 2, axis=(1, 2))
        self.recon_min_ = float(np.min(train_recon_scores))
        self.recon_max_ = float(np.max(train_recon_scores))

        padded_scores = self._window_scores(train_windows)
        _, starts, padded_len = self._make_windows(X_scaled)
        self.decision_scores_ = self._pointwise_scores(
            padded_scores,
            starts,
            padded_len,
            len(X),
        )
        self.decision_scores_ = self._sharpen_scores(self.decision_scores_)
        return self

    def decision_function(self, X):
        if self.scaler is None or self.autoencoder is None or self.encoder is None:
            raise RuntimeError("MyModel must be fitted before calling decision_function.")

        X = _to_2d_array(X)
        X_scaled = self._scale_transform(X)

        # [Improvement 3] Adaptive stride — balance speed vs resolution
        n = len(X_scaled)
        if n <= 5000:
            inf_stride = 1
        elif n <= 50000:
            inf_stride = 3
        else:
            inf_stride = 5
        windows, starts, padded_len = self._make_windows(X_scaled, stride=inf_stride)
        scores = self._window_scores(windows)
        point_scores = self._pointwise_scores(scores, starts, padded_len, len(X))

        # [Improvement 5] Sharpen scores to amplify anomaly peaks
        point_scores = self._sharpen_scores(point_scores)

        return point_scores

    def decision_function_detailed(self, X):
        """Return combined + individual component scores for dashboard breakdown.

        Returns
        -------
        dict with keys: 'combined', 'reconstruction', 'prototype'
            Each value is a 1-D numpy array of point-wise anomaly scores.
        """
        if self.scaler is None or self.autoencoder is None or self.encoder is None:
            raise RuntimeError("MyModel must be fitted before calling decision_function_detailed.")

        X = _to_2d_array(X)
        X_scaled = self._scale_transform(X)

        n = len(X_scaled)
        if n <= 5000:
            inf_stride = 1
        elif n <= 50000:
            inf_stride = 3
        else:
            inf_stride = 5

        windows, starts, padded_len = self._make_windows(X_scaled, stride=inf_stride)
        combined_w, recon_w, proto_w = self._window_scores_detailed(windows)

        combined_pt = self._pointwise_scores(combined_w, starts, padded_len, len(X))
        recon_pt = self._pointwise_scores(recon_w, starts, padded_len, len(X))
        proto_pt = self._pointwise_scores(proto_w, starts, padded_len, len(X))

        combined_pt = self._sharpen_scores(combined_pt)
        recon_pt = self._sharpen_scores(recon_pt)
        proto_pt = self._sharpen_scores(proto_pt)

        return {
            "combined": combined_pt,
            "reconstruction": recon_pt,
            "prototype": proto_pt,
        }