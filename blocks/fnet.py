from __future__ import annotations

import tensorflow as tf
import tensorflow_addons as tfa


class FNetLayer(tf.keras.layers.Layer):
    def __init__(self, num_patches: int, embedding_dim: int, dropout_rate: float, *args, **kwargs) -> FNetLayer:
        super().__init__(*args, **kwargs)

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=embedding_dim),
                tfa.layers.GELU(),
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Apply fourier transformations.
        x = tf.cast(
            tf.signal.fft2d(tf.cast(inputs, dtype=tf.dtypes.complex64)),
            dtype=tf.dtypes.float32,
        )
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = self.normalize1(x)
        # Apply Feedfowrad network.
        x_ffn = self.ffn(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        return self.normalize2(x)
