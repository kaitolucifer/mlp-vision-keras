import tensorflow as tf


def get_data_augmentation() -> tf.keras.Model:
    return tf.keras.Sequential(
            [
                tf.keras.layers.Normalization(),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
                tf.keras.layers.RandomRotation(factor=0.2),
                tf.keras.layers.RandomContrast(factor=0.2),
            ],
            name="data_augmentation",
        )
