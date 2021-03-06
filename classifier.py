from __future__ import annotations
from typing import Union

import tensorflow as tf
import numpy as np

from layers import MultiHeadSelfAttention, Patch
from blocks import MLPMixerLayer, FNetLayer, gMLPLayer


class Classifier(tf.keras.Model):
    def __init__(self, mlp_block: str, num_blocks: int,
                 embedding_dim: int, dropout_rate: float,
                 patch_size: int, num_patches: int, num_classes: int, augmentation: tf.Keras.Model,
                 positional_encoding: bool = False, self_attention: bool = False, num_heads: int = None) -> Classifier:
        super().__init__()
        self.positional_encoding = positional_encoding
        self.self_attention = self_attention

        self.augmentation = augmentation

        self.patches = Patch(patch_size, num_patches)
        self.dense = tf.keras.layers.Dense(units=embedding_dim)

        if self.positional_encoding:
            positions = tf.range(start=0, limit=num_patches, delta=1)
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=embedding_dim
            )(positions)

        self.blocks = []
        match mlp_block:
            case "mlp-mixer":
                for _ in range(num_blocks):
                    if self.self_attention:
                        self.blocks += [MLPMixerLayer(num_patches, embedding_dim, dropout_rate), MultiHeadSelfAttention(embedding_dim, num_heads)]
                    else:
                        self.blocks += [MLPMixerLayer(num_patches, embedding_dim, dropout_rate)]
            case "fnet":
                for _ in range(num_blocks):
                    if self.self_attention:
                        self.blocks += [FNetLayer(num_patches, embedding_dim, dropout_rate), MultiHeadSelfAttention(embedding_dim, num_heads)]
                    else:
                        self.blocks += [FNetLayer(num_patches, embedding_dim, dropout_rate)]
            case "gmlp":
                for _ in range(num_blocks):
                    self.blocks += [gMLPLayer(num_patches, embedding_dim, dropout_rate, self.self_attention, num_heads)]

        self.blocks = tf.keras.Sequential(self.blocks)

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.augmentation(inputs)
        x = self.patches(x)
        x = self.dense(x)
        if self.positional_encoding:
            x = x + self.position_embedding
        x = self.blocks(x)
        x = self.global_avg_pool(x)  # output: [batch_size, embedding_dim]
        x = self.dropout(x)
        return self.classifier(x)

    def fit(self, x: Union[np.ndarray, tf.data.Dataset] = None, y:  np.ndarray = None, batch_size: int = None, epochs: int = 1,
            validation_split: float = 0.0, callbacks: list[tf.keras.callbacks.Callback] = None, **kwargs) -> tf.keras.callbacks.History:
        if not y:
            x_ds = x.map(lambda x, y: x)
            self.augmentation.layers[0].adapt(x_ds)
        else:
            self.augmentation.layers[0].adapt(x)
        return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                           validation_split=validation_split, callbacks=callbacks, **kwargs)
