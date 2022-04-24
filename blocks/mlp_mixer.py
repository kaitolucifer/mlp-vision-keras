from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


class MLPMixerLayer(keras.layers.Layer):
    def __init__(self, num_patches: int, hidden_units: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                keras.layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                keras.layers.Dense(units=num_patches),
                keras.layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                keras.layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                keras.layers.Dense(units=hidden_units),
                keras.layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x
