from __future__ import annotations

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, *args, **kwargs) -> SelfAttention:
        super().__init__(*args, **kwargs)

        self.dense = tf.keras.layers.Dense(units=embedding_dim)
        self.wq = tf.keras.layers.Dense(units=embedding_dim)
        self.wk = tf.keras.layers.Dense(units=embedding_dim)
        self.wv = tf.keras.layers.Dense(units=embedding_dim)
        self.dense = tf.keras.layers.Dense(units=embedding_dim)

    def scaled_dot_product_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        # matmul_qk: (batch_size, num_patches, num_patches)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # attention_weights: (batch_size, num_patches, num_patches)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # output: (batch_size, num_patches, embedding_dim)
        output = tf.matmul(attention_weights, v)
        return output

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        q, k, v = tf.split(inputs, num_or_size_splits=3, axis=-1)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        x = self.scaled_dot_product_attention(q, k, v)
        x = self.dense(x)
        return x
