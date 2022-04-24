from __future__ import annotations

import tensorflow as tf


class Patch(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, num_patches: int) -> Patch:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(
            patches, [batch_size, self.num_patches, patch_dims])
        return patches
