from __future__ import annotations

import tensorflow as tf
import tensorflow_addons as tfa

from self_attention import SelfAttention


class gMLPLayer(tf.keras.layers.Layer):
    def __init__(self, num_patches: int, embedding_dim: int, dropout_rate: float, self_attention: bool = False, *args, **kwargs) -> gMLPLayer:
        super().__init__(*args, **kwargs)

        self.channel_projection1 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=embedding_dim * 2),
                tfa.layers.GELU(),
                tf.keras.layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = tf.keras.layers.Dense(units=embedding_dim)

        self.spatial_projection = tf.keras.layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.attention = SelfAttention(embedding_dim) if self_attention else None


    def spatial_gating_unit(self, x: tf.Tensor) -> tf.Tensor:
        # Split x along the channel dimensions.
        # Tensors u and v will in th shape of [batch_size, num_patchs, embedding_dim].
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = tf.linalg.matrix_transpose(v)
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.linalg.matrix_transpose(v_projected)
        if self.attention:
            attention_output = self.attention(x)
            v_projected += attention_output
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected
