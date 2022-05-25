from __future__ import annotations

import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, num_heads: int, *args, **kwargs) -> MultiHeadSelfAttention:
        super().__init__(*args, **kwargs)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth = embedding_dim // num_heads
    
        self.wq = tf.keras.layers.Dense(units=embedding_dim)
        self.wk = tf.keras.layers.Dense(units=embedding_dim)
        self.wv = tf.keras.layers.Dense(units=embedding_dim)
        self.dense = tf.keras.layers.Dense(units=embedding_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, num_patches, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        # matmul_qk: (batch_size, num_heads, num_patches, num_patches)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # attention_weights: (batch_size, num_heads, num_patches, num_patches)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # output: (batch_size, num_heads, num_patches, embedding_dim)
        output = tf.matmul(attention_weights, v)
        return output

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        q, k, v = tf.split(inputs, num_or_size_splits=3, axis=-1)
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, num_patches, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        x = self.scaled_dot_product_attention(q, k, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.embedding_dim))  # (batch_size, num_patches, embedding_dim)
        x = self.dense(x)
        return x